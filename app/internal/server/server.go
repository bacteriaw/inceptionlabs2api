// Package server exposes the OpenAI-compatible HTTP surface: chat
// completions (stream and non-stream), models, and health.
package server

import (
	"context"
	"crypto/subtle"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"inceptionlabs2api/internal/clear"
	"inceptionlabs2api/internal/openai"
	"inceptionlabs2api/internal/pool"
	"inceptionlabs2api/internal/tools"
	"inceptionlabs2api/internal/upstream"
)

// maxBodyBytes caps request bodies before JSON decoding. It exists only to
// bound memory use and is far above any request the upstream itself accepts;
// it is not a substitute for the upstream's context limit (SPEC §1.1).
const maxBodyBytes = 6 << 20

// Chatter is the upstream abstraction the handlers depend on.
type Chatter interface {
	Chat(ctx context.Context, req *upstream.ChatRequest, idle time.Duration, onMalformed func(string)) (*upstream.Stream, error)
}

// Searcher is the standalone retrieval abstraction. SearchConfig is called at
// most once per process to expose search.available on /healthz; its failure
// is diagnostic only and never disables the endpoint.
type Searcher interface {
	Search(ctx context.Context, req *upstream.SearchRequest, idle time.Duration) (*upstream.SearchResult, error)
	SearchConfig(ctx context.Context) (*upstream.SearchConfigResponse, error)
}

// PoolStats reports pool size for /healthz.
type PoolStats interface {
	Stats() pool.Stats
}

// ClearanceStats reports checkpoint clearance state for /healthz. A nil
// source means the gateway does not track clearance (e.g. unit tests).
type ClearanceStats interface {
	State() clear.State
}

// Config configures the HTTP surface.
type Config struct {
	APIKey      string
	Model       string
	Models      []string
	IdleTimeout time.Duration
	// MaxMessages caps one chat request's message count. Zero means 96.
	// The upstream rejects >100 messages with "This conversation is too long".
	MaxMessages int
	// EmptyRetries is how many times an empty upstream answer (no text at all)
	// is retried before reporting 502 upstream_empty_response. Zero keeps the
	// single attempt behaviour; negative is treated as zero.
	EmptyRetries int
	// CompactKeepRecent keeps this many newest messages structurally intact when
	// history is folded to fit MaxMessages. Zero means MaxMessages-8.
	CompactKeepRecent int
	// SearchTimeout bounds /v1/web_search's sliding idle watchdog. Zero means
	// IdleTimeout.
	SearchTimeout time.Duration
	// SearchProbeTimeout bounds the one startup /api/search-lab/config probe.
	// Zero means 15s.
	SearchProbeTimeout time.Duration
	// AudioTimeout bounds the pre-header phase of one /api/tts or /api/stt
	// call. Zero means 120s.
	AudioTimeout time.Duration
	// AudioMaxBytes caps a transcription upload. Zero means 25 MiB.
	AudioMaxBytes int64
	// DecisionTimeout bounds one whole /api/decide/run call. Zero means 120s.
	DecisionTimeout time.Duration
	// PreambleStyle selects the tool preamble skeleton ("hermes" or
	// "inception"). Empty keeps the measured hermes default.
	PreambleStyle string
	// NativeWebSearch auto-declares web_search on tool-bearing chat requests
	// and serves its calls from the attached Searcher. A request that already
	// declares web_search, or a server without a Searcher, keeps the tool off.
	NativeWebSearch bool
	// Logf is optional; nil means log.Printf.
	Logf func(format string, args ...any)
}

// Counters is a snapshot of request accounting for /healthz.
type Counters struct {
	Requests  uint64 `json:"requests"`
	Streams   uint64 `json:"streams"`
	Errors    uint64 `json:"errors"`
	Upstream  uint64 `json:"upstream_errors"`
	TTS       uint64 `json:"tts"`
	STT       uint64 `json:"stt"`
	Decisions uint64 `json:"decisions"`
}

// Server holds the handler dependencies.
type Server struct {
	cfg    Config
	up     Chatter
	se     Searcher
	pool   PoolStats
	clr    ClearanceStats
	audio  Speech
	decide Decider
	now    func() time.Time
	logf   func(format string, args ...any)

	searchMu sync.Mutex
	// searchAvailable is nil until the config probe succeeds; false is a
	// definitive "the deployment disabled retrieval".
	searchAvailable *bool
	probeOnce       sync.Once

	requests      atomic.Uint64
	streams       atomic.Uint64
	errors        atomic.Uint64
	upErrors      atomic.Uint64
	tts           atomic.Uint64
	stt           atomic.Uint64
	decisionCalls atomic.Uint64
}

// New builds a Server. Models defaults to [Model].
func New(cfg Config, up Chatter, p PoolStats) *Server {
	if len(cfg.Models) == 0 {
		cfg.Models = []string{cfg.Model}
	}
	// Wire the configured preamble skeleton into the tools package once at
	// boot; empty keeps the measured hermes default.
	if cfg.PreambleStyle != "" {
		tools.PreStyle = cfg.PreambleStyle
	}
	if cfg.IdleTimeout <= 0 {
		cfg.IdleTimeout = 60 * time.Second
	}
	if cfg.SearchTimeout <= 0 {
		cfg.SearchTimeout = cfg.IdleTimeout
	}
	if cfg.SearchProbeTimeout <= 0 {
		cfg.SearchProbeTimeout = 15 * time.Second
	}
	logf := cfg.Logf
	if logf == nil {
		logf = log.Printf
	}
	return &Server{cfg: cfg, up: up, pool: p, now: time.Now, logf: logf}
}

// SetSearcher attaches the standalone retrieval backend. Without one the
// route answers 503 rather than panicking.
func (s *Server) SetSearcher(se Searcher) { s.se = se }

// SetClearance attaches the clearance state reported by /healthz.
func (s *Server) SetClearance(c ClearanceStats) { s.clr = c }

// StartSearchProbe launches the one-shot /api/search-lab/config probe in the
// background. It is a no-op when no searcher is attached, and repeated calls
// probe only once.
func (s *Server) StartSearchProbe() {
	s.probeOnce.Do(func() {
		if s.se == nil {
			return
		}
		go s.probeSearch(context.Background())
	})
}

// probeSearch records the deployment's retrieval capability. A failed probe
// leaves the value unknown (null) — it never disables the endpoint, because
// the config route can be challenged independently of the run route.
func (s *Server) probeSearch(ctx context.Context) {
	if s.se == nil {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, s.cfg.SearchProbeTimeout)
	defer cancel()
	cfg, err := s.se.SearchConfig(ctx)
	if err != nil {
		s.logf("search: config probe failed (search stays enabled, availability unknown): %v", err)
		return
	}
	if cfg == nil {
		s.logf("search: config probe returned no body; availability unknown")
		return
	}
	available := cfg.RetrievalAvailable
	s.searchMu.Lock()
	s.searchAvailable = &available
	s.searchMu.Unlock()
	s.logf("search: retrievalAvailable=%v", available)
}

// SearchAvailable returns the cached probe result, nil when unknown. The probe
// is advisory: it is reported on /healthz and never gates a request, so a stale
// or wrong value cannot take a working endpoint offline (SPEC §7.1).
func (s *Server) SearchAvailable() *bool {
	s.searchMu.Lock()
	defer s.searchMu.Unlock()
	if s.searchAvailable == nil {
		return nil
	}
	v := *s.searchAvailable
	return &v
}

// Handler returns the routed http.Handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.chatCompletions)
	mux.HandleFunc("/v1/web_search", s.webSearch)
	mux.HandleFunc("/v1/audio/speech", s.speech)
	mux.HandleFunc("/v1/audio/transcriptions", s.transcription)
	mux.HandleFunc("/v1/decisions", s.decisions)
	mux.HandleFunc("/v1/models", s.models)
	mux.HandleFunc("/healthz", s.healthz)
	return cors(s.authenticate(mux))
}

// cors answers browser preflights and lets browser-based clients call the
// gateway. The API is keyed, not cookie-based, so a permissive origin is safe
// and avoids a configuration matrix nobody needs. It wraps auth because a
// browser preflight never carries credentials: answering 401 to OPTIONS would
// make the documented CORS support unusable whenever a key is set.
func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", "*")
		h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Api-Key")
		h.Set("Access-Control-Max-Age", "86400")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// authenticate enforces the optional bearer key. /healthz is always open.
func (s *Server) authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.APIKey == "" || r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}
		provided := ""
		if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
			provided = strings.TrimPrefix(h, "Bearer ")
		} else if h := r.Header.Get("X-Api-Key"); h != "" {
			provided = h
		}
		if subtle.ConstantTimeCompare([]byte(provided), []byte(s.cfg.APIKey)) != 1 {
			openai.WriteError(w, openai.New(http.StatusUnauthorized, "invalid_request_error", openai.CodeInvalidAPIKey, "",
				"invalid API key; send Authorization: Bearer <key>"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	s.requests.Add(1)

	body, re := s.readJSONBody(w, r)
	if re != nil {
		s.fail(w, re)
		return
	}
	tag := wireTag()
	wireDumpRequest(tag, body)
	req, re := openai.Decode(body)
	if re != nil {
		s.fail(w, re)
		return
	}
	// 上游硬上限 100 条消息（101 条即 "This conversation is too long"），
	// 与 token 大小无关。超限时先把最旧的部分折叠成一条转录消息（文本不丢），
	// 仍然超限才裁掉最旧的历史。
	if compacted, folded := compactMessages(req.Messages, s.cfg.CompactKeepRecent, s.cfg.MaxMessages); folded > 0 {
		s.logf("chat: compacted %d oldest messages into one transcript bubble (%d -> %d)", folded, len(req.Messages), len(compacted))
		req.Messages = compacted
	}
	if trimmed, dropped := trimMessages(req.Messages, s.cfg.MaxMessages); dropped > 0 {
		s.logf("chat: trimmed %d oldest messages (%d -> %d) for the upstream 100-message cap", dropped, len(req.Messages), len(trimmed))
		req.Messages = trimmed
	}
	if !s.knownModel(req.Model) {
		s.fail(w, openai.New(http.StatusNotFound, "invalid_request_error", openai.CodeModelNotFound, "model",
			fmt.Sprintf("model %q does not exist; available models: %s", req.Model, strings.Join(s.cfg.Models, ", "))))
		return
	}
	// Tool-bearing requests take §4.1's path, which normalises history with
	// tool payloads intact. A nil plan means the legacy one, and it must stay
	// byte-identical for every request without tools.
	plan, re := planTools(req, s.nativeSearchTool())
	if re != nil {
		s.fail(w, re)
		return
	}
	if plan != nil {
		s.chatWithTools(w, r, req, plan)
		return
	}

	msgs, _, re := openai.NormalizeMessages(req.Messages)
	if re != nil {
		s.fail(w, re)
		return
	}

	ureq := upstream.NewChatRequest(msgs, req.ReasoningEffort)
	if req.Stream {
		s.stream(w, r, ureq, req.Model)
		return
	}
	s.complete(w, r, ureq, req.Model)
}

func (s *Server) knownModel(model string) bool {
	for _, m := range s.cfg.Models {
		if m == model {
			return true
		}
	}
	return false
}

// fail records and writes a client-facing error.
func (s *Server) fail(w http.ResponseWriter, re *openai.RequestError) {
	if re.Status >= 500 {
		s.errors.Add(1)
	}
	openai.WriteError(w, re)
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		methodNotAllowed(w, "GET, HEAD")
		return
	}
	created := s.now().Unix()
	list := openai.ModelList{Object: "list", Data: make([]openai.Model, 0, len(s.cfg.Models))}
	for _, m := range s.cfg.Models {
		list.Data = append(list.Data, openai.Model{ID: m, Object: "model", Created: created, OwnedBy: "inceptionlabs"})
	}
	writeJSON(w, http.StatusOK, list)
}

// healthz reports liveness, pool size, and request counters. It is never
// behind auth, so orchestrators can probe it.
func (s *Server) healthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		methodNotAllowed(w, "GET, HEAD")
		return
	}
	body := map[string]any{
		"ok": true,
		"counters": Counters{
			Requests:  s.requests.Load(),
			Streams:   s.streams.Load(),
			Errors:    s.errors.Load(),
			Upstream:  s.upErrors.Load(),
			TTS:       s.tts.Load(),
			STT:       s.stt.Load(),
			Decisions: s.decisionCalls.Load(),
		},
		"search": map[string]any{"available": s.SearchAvailable()},
	}
	if s.pool != nil {
		body["pool"] = s.pool.Stats()
	}
	if s.clr != nil {
		body["clearance"] = s.clr.State()
	}
	writeJSON(w, http.StatusOK, body)
}
