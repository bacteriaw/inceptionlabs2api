package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"inceptionlabs2api/internal/openai"
	"inceptionlabs2api/internal/tools"
	"inceptionlabs2api/internal/upstream"
)

// toolPlan is the request-scoped state derived from tools/tool_choice.
type toolPlan struct {
	manifest *tools.Manifest
	hist     []openai.History
	// native reports that the gateway's own web_search tool was added to the
	// manifest, so its calls must be served internally rather than emitted.
	native bool
	// folded reports that hist was folded into a transcript bubble, so the
	// preamble carries the history-format teaching section.
	folded bool
}

// planTools builds the manifest and history for a tool-bearing request. It
// returns a nil plan when the request must take the legacy path. native is the
// optional gateway-served search tool; it is added only when the client did
// not declare that name itself (a client tool always wins).
//
// A request with NO client tools also takes the tools path when the native
// search tool is available (SDD-refusal-recovery T2): a plain chat turn has
// no preamble and no search guidance otherwise, so time-sensitive questions
// get answered from memory alone. The gateway then declares its own
// web_search, serves the calls internally, and the client sees only the
// final grounded answer. tool_choice:"none" is the client's explicit opt-out
// and keeps the legacy path byte-identical; so does every configuration
// where nativeSearchTool() is nil (NativeWebSearch off, no searcher).
func planTools(req *openai.ChatRequest, native *tools.Tool) (*toolPlan, *openai.RequestError) {
	if req.ToolChoice.Kind == openai.ToolChoiceNone {
		return nil, nil
	}
	if len(req.Tools) == 0 {
		if native == nil {
			return nil, nil
		}
		// The auto-declared manifest carries only the gateway's search tool
		// with the default auto choice, so the model may answer in prose when
		// memory suffices and calls web_search only when it does not.
		return buildPlan(req, []tools.Tool{*native}, tools.Choice{Auto: true}, false, true)
	}
	list := make([]tools.Tool, 0, len(req.Tools)+1)
	for _, t := range req.Tools {
		if t.Function.Name == tools.NativeSearchName {
			native = nil
		}
		list = append(list, tools.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters})
	}
	if native != nil {
		list = append(list, *native)
	}
	choice := tools.Choice{}
	switch req.ToolChoice.Kind {
	case openai.ToolChoiceRequired:
		choice.Required = true
	case openai.ToolChoiceNamed:
		choice.Named = req.ToolChoice.Name
	default:
		choice.Auto = true
	}
	parallelOff := req.ParallelToolCalls != nil && !*req.ParallelToolCalls
	return buildPlan(req, list, choice, parallelOff, native != nil)
}

// buildPlan renders the manifest and folds the history for one request,
// shared by the client-tools and the native-only shapes. isNative reports
// whether the gateway's web_search is in the list, so its calls are served
// internally rather than emitted.
func buildPlan(req *openai.ChatRequest, list []tools.Tool, choice tools.Choice, parallelOff, isNative bool) (*toolPlan, *openai.RequestError) {
	manifest, err := tools.BuildManifest(list, choice, parallelOff)
	if err != nil {
		return nil, openai.Invalid(err.Error())
	}
	hist, _, re := openai.NormalizeHistory(req.Messages)
	if re != nil {
		return nil, re
	}
	// Fold before any upstream conversion and exactly once per client
	// request: the native loop and every recovery retry append to the
	// already-folded history, so folding never re-runs and the bubble never
	// grows a second layer. The manifest preamble occupies one upstream
	// slot, so it counts toward the fold trigger.
	hist, folded := openai.FoldHistory(hist, true)
	if folded {
		// Additive preamble protocol (wire-format law: the measured skeleton
		// sentences stay byte-identical; the section is appended only when
		// folding fires, so unfolded requests keep today's preamble).
		manifest.Preamble += "\n\n" + openai.HistoryFormatSection + "\n"
	}
	return &toolPlan{manifest: manifest, hist: hist, native: isNative, folded: folded}, nil
}

// chatWithTools answers a request whose tools are active. A plain manifest
// streams; the native web_search loop resolves internally first, so a
// manifest with native search on takes the pre-buffered path.
func (s *Server) chatWithTools(w http.ResponseWriter, r *http.Request, req *openai.ChatRequest, plan *toolPlan) {
	ureq := upstream.NewToolChatRequestHistory(plan.hist, req.ReasoningEffort, plan.manifest)
	if req.Stream {
		s.streamTools(w, r, ureq, req.Model, plan.manifest, plan)
		return
	}
	s.completeTools(w, r, ureq, req.Model, plan.manifest, plan)
}

// round is one upstream response with its parsing applied.
type round struct {
	content   string
	reasoning string
	calls     []tools.Call
	cites     upstream.CitationCollector
	// nativeCites carries the sources the native search loop retrieved, in
	// first-seen order, deduped by URL. Index is assigned in annotations()
	// so chat-path citations and search sources stay in one 1-based space.
	nativeCites []openai.Citation
}

// annotations returns the citations the client-facing answer carries: the
// chat stream's source-url citations plus, on the native path, the search
// loop's retrieved sources. On the native path the chat collector stays
// empty (webSearchEnabled is false with a manifest), and vice versa, so the
// two families never overlap in practice — the merge is order-preserving
// regardless.
func (r *round) annotations() []openai.Citation {
	base := r.cites.Citations()
	if len(r.nativeCites) == 0 {
		return base
	}
	out := make([]openai.Citation, 0, len(base)+len(r.nativeCites))
	out = append(out, base...)
	for i, c := range r.nativeCites {
		c.Index = len(base) + i + 1
		out = append(out, c)
	}
	return out
}

// readRound opens one upstream stream and consumes it fully.
func (s *Server) readRound(ctx context.Context, ureq *upstream.ChatRequest, manifest *tools.Manifest) (*round, error) {
	stream, err := s.up.Chat(ctx, ureq, s.cfg.IdleTimeout, func(line string) {
		s.logf("upstream: skipping malformed SSE line: %q", truncate(line, 200))
	})
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var out round
	parser := tools.NewParser(manifest.Declared)
	rtag := wireTag()
	readErr := stream.Reader.Read(ctx, func(ev upstream.Event) error {
		out.cites.Observe(ev, s.logf)
		if wireDebugOn() {
			wireDumpEvent(rtag, ev, 0, 0)
		}
		switch ev.Type {
		case upstream.EventReasoningDelta:
			out.reasoning += ev.Reasoning()
		case upstream.EventTextStart, upstream.EventTextDelta:
			out.content += string(parser.Feed([]byte(ev.ContentText())))
		}
		return nil
	})
	if readErr != nil {
		return nil, readErr
	}
	out.content += string(parser.Flush())
	out.calls = parser.Calls()
	return &out, nil
}

// emptyToolAnswer reports the measured upstream failure mode: with a tool
// manifest present the model occasionally returns reasoning-only, zero text
// and zero calls (probe/outQ q1/q3/q4, probe/outP2 timing runs). The upstream
// web surface shows the same flakiness without any preamble, so this is the
// model's occasional refusal-shaped pause, not a request defect. Reasoning
// without text or a call is the pause itself (the streaming path always
// treated it so via sawAlive), so only content and calls decide. The client
// asked for tool calling; an empty answer is not a valid tool response, so
// the gateway re-asks once internally with the same manifest — protocol
// completion, not content interception: whatever the retry returns is passed
// through verbatim, empty or not.
func (r *round) emptyToolAnswer() bool {
	return r.content == "" && len(r.calls) == 0
}

// retryKind classifies one zero-call round for the recovery retry. The empty
// answer keeps its own legacy branch (same-manifest re-ask); a broken-call
// round — truncated envelope or unparseable block — gets the ToolForge-style
// recovery retry; clean prose returns false: the gateway translates, never
// blocks, and a clean prose answer is a legal terminal.
func (s *Server) retryKind(res *round, calls []tools.Call) (tools.RecoveryKind, bool) {
	all := res.calls
	if calls != nil {
		all = calls
	}
	if len(all) > 0 {
		return tools.RecoveryClean, false
	}
	if res.emptyToolAnswer() {
		return tools.RecoveryClean, true // legacy empty branch
	}
	kind := tools.ClassifyNoCall(res.content, tools.ForeignOpenMarker)
	s.logf("tools: [recovery kind=%s] zero-call round (%d content bytes)", kind, len(res.content))
	return kind, kind != tools.RecoveryClean
}

// recoveryRequest clones ureq with the broken round appended as history —
// assistant(original content) + user(retry message) — so the model sees what
// it emitted and is asked to re-emit it properly.
func recoveryRequest(ureq *upstream.ChatRequest, res *round, message string) *upstream.ChatRequest {
	clone := *ureq
	msgs := make([]upstream.Message, 0, len(ureq.Messages)+2)
	msgs = append(msgs, ureq.Messages...)
	if res.content != "" {
		msgs = append(msgs, upstream.Message{
			Parts: []upstream.Part{{Type: "text", Text: res.content, State: "done"}},
			ID:    upstream.RandomID("m"),
			Role:  "assistant",
		})
	}
	msgs = append(msgs, upstream.Message{
		Parts: []upstream.Part{{Type: "text", Text: message}},
		ID:    upstream.RandomID("m"),
		Role:  "user",
	})
	clone.Messages = msgs
	return &clone
}

// recoveryRequestScrubbed is recoveryRequest with the switch-note scrub: on
// the REFUSAL retry only, the model-switch note is stripped from the
// historical user turns too, not just the retry re-anchor. The note is the
// measured sole trigger of the canned identity refusal (il2a-live2: with it
// the model refuses twice, identical request minus it answers normally), so
// a retry that leaves it in history re-fires the reflex and burns the one
// bounded retry the budget allows. Scrubbing is retry-scoped: the clone is
// a gateway-owned request and the client's own history is never touched.
func recoveryRequestScrubbed(ureq *upstream.ChatRequest, res *round, message string) *upstream.ChatRequest {
	clone := *ureq
	msgs := make([]upstream.Message, len(ureq.Messages), len(ureq.Messages)+2)
	for i, m := range ureq.Messages {
		if m.Role == "user" {
			scrubbed := false
			parts := make([]upstream.Part, len(m.Parts))
			for j, p := range m.Parts {
				if p.Type == "text" {
					t := tools.StripSwitchNote(p.Text)
					if t != p.Text {
						scrubbed = true
					}
					parts[j] = p
					parts[j].Text = t
				} else {
					parts[j] = p
				}
			}
			if scrubbed {
				m = upstream.Message{Parts: parts, ID: m.ID, Role: m.Role}
			}
		}
		msgs[i] = m
	}
	clone.Messages = msgs
	clone.ID = upstream.RandomID("c")
	return recoveryRequest(&clone, res, message)
}

// recoverRound runs the ONE bounded recovery retry for a retry-worthy
// zero-call round: the empty answer re-asks with the same manifest (legacy
// behavior), the broken-call kinds re-ask with the original output appended
// as history plus the retry message. The retry result passes through
// verbatim — call or prose, empty or not; a transport failure keeps the
// ORIGINAL round rather than failing a client who already has an answer.
func (s *Server) recoverRound(ctx context.Context, ureq *upstream.ChatRequest, manifest *tools.Manifest, res *round, kind tools.RecoveryKind) *round {
	retryReq := recoveryRequestFor(ureq, res, kind)
	s.logf("tools: %s, retrying once", retryLogReason(kind, res))
	retry, err := s.readRound(ctx, retryReq, manifest)
	if err != nil {
		s.upErrors.Add(1)
		s.logf("tools: recovery retry failed: %v (passing original round through)", err)
		return res
	}
	return retry
}

// retryLogReason names the retry for the log line.
func retryLogReason(kind tools.RecoveryKind, res *round) string {
	if res.emptyToolAnswer() {
		return "empty answer round (no text/call)"
	}
	return fmt.Sprintf("[recovery kind=%s] broken-call round", kind)
}

// recoveryRequestFor builds the ureq a recovery retry re-asks with, shared by
// every retry path (stream/non-stream × plain/native). The empty answer
// keeps the same manifest (legacy); the broken-call kinds carry the original
// output appended as history plus the retry message.
func recoveryRequestFor(ureq *upstream.ChatRequest, res *round, kind tools.RecoveryKind) *upstream.ChatRequest {
	switch {
	case res.emptyToolAnswer():
		return ureq
	case kind == tools.RecoveryTruncated:
		return recoveryRequest(ureq, res, tools.RetryMessageTruncated(res.content))
	case kind == tools.RecoveryParseFailed:
		return recoveryRequest(ureq, res, tools.RetryMessageParseFailed(res.content))
	case kind == tools.RecoveryRefused:
		// recoveryRequest appends the refused round's original content as the
		// assistant turn — the model must see what it emitted — while the
		// refusal-retry message carries the re-instruction (the canned shape
		// is to be dropped, not repaired). The scrubbed variant strips the
		// switch note from the historical user turns: leaving it in re-fires
		// the identity reflex and burns the one retry.
		return recoveryRequestScrubbed(ureq, res, tools.RetryMessageRefusal(res.content))
	}
	return ureq
}

// nativeRetryHistory builds the history a native recovery retry re-enters the
// loop with: the accumulated history plus, for the broken-call kinds, the
// broken round and the retry message as History turns — the same append
// recoveryRequest performs on the wire, done at the history level because
// resolveNative owns its own history threading and builds every round from
// the history it was given, not from a request's message list. The empty
// answer keeps the accumulated history unchanged (same-manifest re-ask).
func nativeRetryHistory(hist []openai.History, res *round, kind tools.RecoveryKind) []openai.History {
	var message string
	scrub := false
	switch {
	case res.emptyToolAnswer():
		return hist
	case kind == tools.RecoveryTruncated:
		message = tools.RetryMessageTruncated(res.content)
	case kind == tools.RecoveryParseFailed:
		message = tools.RetryMessageParseFailed(res.content)
	case kind == tools.RecoveryRefused:
		// Same shape as the truncated branch: the refused round's content is
		// non-empty and must ride the history verbatim, followed by the
		// refusal-retry instruction. The switch note is scrubbed from
		// historical user turns too: it is the measured sole trigger of the
		// identity reflex (il2a-live2) and leaving it in re-fires the
		// refusal on the retry the budget allows.
		message = tools.RetryMessageRefusal(res.content)
		scrub = true
	default:
		return hist
	}
	out := make([]openai.History, 0, len(hist)+2)
	for _, h := range hist {
		if scrub && h.Role == "user" && len(h.Results) == 0 {
			ts := make([]string, len(h.Texts))
			changed := false
			for i, t := range h.Texts {
				ts[i] = tools.StripSwitchNote(t)
				changed = changed || ts[i] != t
			}
			if changed {
				h = openai.History{Role: h.Role, Texts: ts, Calls: h.Calls, Results: h.Results}
			}
		}
		out = append(out, h)
	}
	if res.content != "" {
		out = append(out, openai.History{Role: "assistant", Texts: []string{res.content}})
	}
	out = append(out, openai.History{Role: "user", Texts: []string{message}})
	return out
}

// emitToolCalls writes one delta frame per parsed call and returns the number
// emitted so far.
func emitToolCalls(w http.ResponseWriter, flusher http.Flusher, calls []tools.Call, emitted int, write func(openai.Delta, *string) bool) int {
	for ; emitted < len(calls); emitted++ {
		c := calls[emitted]
		delta := openai.Delta{ToolCalls: []openai.ToolCallDelta{{
			Index: emitted,
			ID:    tools.NewID(),
			Type:  "function",
			Function: openai.FunctionCallDelta{
				Name:      c.Name,
				Arguments: c.Arguments,
			},
		}}}
		if !write(delta, nil) {
			return emitted
		}
	}
	return emitted
}

// setStreamHeaders commits the SSE response headers shared by every streaming
// path (plain chat, tools, native). Headers are set only after the upstream
// stream is open and the round(s) are classified, so early failures still
// carry a proper HTTP status.
func setStreamHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
}

// streamTools is the streaming path for a plain manifest. With the
// native search tool active it runs the internal loop first: every round is
// consumed privately (no bytes) until the model stops calling web_search, and
// only that final round is replayed as frames.
//
// The plain path consumes each round fully before any frame leaves (the
// upstream sends whole answers in single events on this diffusion model, so
// nothing is lost). That is what makes the recovery retry sound: a zero-call
// round can be classified — clean prose, the measured empty answer, a
// truncated call envelope, an unparseable block — BEFORE the client has seen
// a byte, so a broken round never streams half its narration and the retry
// result (call or prose) is passed through verbatim.
//
// The SSE headers and the role frame are committed only when the final round
// is replayed: a round that is still empty after the recovery retry is
// exhausted returns a real HTTP 502 instead of a 200 with an empty SSE
// stream, so the client can treat the empty answer as an HTTP error.
func (s *Server) streamTools(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string, manifest *tools.Manifest, plan *toolPlan) {
	if plan.native {
		s.streamNative(w, r, ureq, model, plan)
		return
	}
	rtag := wireTag()
	res, err := s.readRound(r.Context(), ureq, manifest)
	if err != nil {
		s.fail(w, s.upstreamError(err))
		return
	}
	if kind, retry := s.retryKind(res, nil); retry {
		res = s.recoverRound(r.Context(), ureq, manifest, res, kind)
	}
	if res.emptyToolAnswer() {
		s.logf("tools: [recovery exhausted] round still empty after retry; returning 502")
		s.fail(w, upstreamEmpty())
		return
	}
	wireDumpRound(rtag, res.content, res.calls)

	flusher, _ := w.(http.Flusher)
	write := s.chunkWriter(w, flusher, upstream.RandomID("chatcmpl"), s.now().Unix(), model, rtag)
	s.replayRound(w, flusher, write, res, res.calls)
}

// streamNative consumes rounds privately until the model stops calling the
// gateway's web_search tool, then replays the final round as frames. Nothing
// from an internal round is ever written, so the first client-visible frame
// is always part of the answer the client asked for. The SSE headers and
// the role frame are committed only when the final round is replayed; a
// final round that is still empty after the recovery retry returns a real
// HTTP 502 (same rule as the plain tools path).
func (s *Server) streamNative(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string, plan *toolPlan) {
	rtag := wireTag()
	res, calls, hist, err := s.resolveNative(r.Context(), ureq, plan, plan.hist, rtag)
	if err != nil {
		if nativeAbort(r, err) {
			return
		}
		s.fail(w, s.upstreamError(err))
		return
	}
	// The final round gets the same classification and one bounded retry as
	// the plain path. Nothing has been written yet, so the retry is free.
	// It re-enters the loop from the ACCUMULATED history plus the recovery
	// turns, so the search the loop already served is never lost.
	if kind, doRetry := s.retryKind(res, calls); doRetry {
		if retryRes, retryCalls, _, err2 := s.resolveNative(r.Context(), ureq, plan, nativeRetryHistory(hist, res, kind), rtag); err2 == nil {
			res, calls = retryRes, retryCalls
		} else if nativeAbort(r, err2) {
			return
		} else {
			s.upErrors.Add(1)
			s.logf("tools: native recovery retry failed: %v (passing original round through)", err2)
		}
	}
	if res.emptyToolAnswer() {
		s.logf("tools: [recovery exhausted] native final round still empty after retry; returning 502")
		s.fail(w, upstreamEmpty())
		return
	}

	flusher, _ := w.(http.Flusher)
	write := s.chunkWriter(w, flusher, upstream.RandomID("chatcmpl"), s.now().Unix(), model, rtag)
	s.replayRound(w, flusher, write, res, calls)
}

// resolveNative runs the internal web_search loop: read a round, serve every
// native call, append its result, and re-ask. startHist is the history the
// loop grows from — plan.hist on the initial call, the accumulated history
// on a recovery retry, so the search turns already served are never lost.
// It returns the first round with no native call plus any client calls
// accumulated from internal rounds (they are emitted with the final answer so
// no client tool call is lost), plus the accumulated history. Every internal
// round is dumped under rtag (env-gated, zero cost off) so the search loop
// is diagnosable from ROUND lines alone. Every source the
// loop retrieved is attached to the returned round as annotations, so the
// client sees the citations the answer's [n] markers refer to even though
// the tool path never fires source-url events.
func (s *Server) resolveNative(ctx context.Context, ureq *upstream.ChatRequest, plan *toolPlan, startHist []openai.History, rtag string) (*round, []tools.Call, []openai.History, error) {
	manifest := plan.manifest
	hist := startHist
	effort := ureq.ReasoningEffort
	budget := 0
	var pending []tools.Call
	var cites sourceCollector
	// One round per allowed search, plus the round that reports the budget
	// exhaustion, plus a final round the model answers in. A model that keeps
	// calling past that is resolved with the calls stripped rather than looped.
	limit := nativeSearchBudget + 2
	var last *round
	for round := 0; round < limit; round++ {
		res, err := s.readRound(ctx, upstream.NewToolChatRequestHistory(hist, effort, manifest), manifest)
		if err != nil {
			return nil, nil, nil, err
		}
		last = res
		wireDumpRound(rtag, res.content, res.calls)
		native, client := nativeRound(res.calls)
		if len(native) == 0 {
			res.nativeCites = cites.cites
			return res, append(pending, client...), hist, nil
		}
		step, used, err := s.serveNativeCalls(ctx, native, res.content, budget, &cites)
		if err != nil {
			return nil, nil, nil, err
		}
		budget = used
		hist = append(hist, step...)
		pending = append(pending, client...)
	}
	// The model ignored the budget and the round cap: answer with its last
	// round, with the unresolved native calls stripped. A refusal-by-loop is
	// not a client-visible error; the stream still closes normally.
	s.logf("tools: native search round cap reached; answering with the last round")
	last.nativeCites = cites.cites
	return last, pending, hist, nil
}

// replayRound writes an already-consumed round as the client-visible answer.
// It owns the delayed header commit: the SSE headers, the streams counter,
// and the role frame are written here — the first client-visible bytes of
// the response — and only for a round that is provably non-empty (the
// callers screen the empty case into a 502 first). The tool_calls finish
// reason is decided by the calls that remain after the native ones were
// served.
func (s *Server) replayRound(w http.ResponseWriter, flusher http.Flusher, write func(openai.Delta, *string) bool, res *round, calls []tools.Call) {
	setStreamHeaders(w)
	s.streams.Add(1)
	if !write(openai.Delta{Role: "assistant"}, nil) {
		return
	}
	if res.reasoning != "" && !write(openai.Delta{ReasoningContent: res.reasoning}, nil) {
		return
	}
	if res.content != "" && !write(openai.Delta{Content: res.content}, nil) {
		return
	}
	emitToolCalls(w, flusher, calls, 0, write)
	finish := "stop"
	if len(calls) > 0 {
		finish = "tool_calls"
	}
	if !write(openai.Delta{Annotations: res.annotations()}, &finish) {
		return
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

// chunkWriter builds the frame writer for one streamed response.
func (s *Server) chunkWriter(w http.ResponseWriter, flusher http.Flusher, id string, created int64, model string, tag string) func(openai.Delta, *string) bool {
	return func(delta openai.Delta, finish *string) bool {
		chunk := openai.Chunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChunkChoice{{Index: 0, Delta: delta, FinishReason: finish}},
		}
		if err := writeEvent(w, chunk); err != nil {
			return false
		}
		wireDumpChunk(tag, chunk)
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
}

// completeTools is the non-streaming path for a plain manifest.
func (s *Server) completeTools(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string, manifest *tools.Manifest, plan *toolPlan) {
	rtag := wireTag()
	var res *round
	var calls []tools.Call
	var err error
	var hist []openai.History
	if plan.native {
		res, calls, hist, err = s.resolveNative(r.Context(), ureq, plan, plan.hist, rtag)
	} else {
		res, err = s.readRound(r.Context(), ureq, manifest)
	}
	if err != nil {
		if errors.Is(err, context.Canceled) && r.Context().Err() != nil {
			return
		}
		s.fail(w, s.upstreamError(err))
		return
	}
	if !plan.native {
		// The native branch already dumped every round (including the
		// final) from inside resolveNative.
		wireDumpRound(rtag, res.content, res.calls)
	}
	if plan.native {
		// The native loop's final round gets the same classification and one
		// bounded retry as the plain path. The retry re-enters the native
		// loop (not a bare readRound) so a retry round that calls web_search
		// is served internally rather than leaked to the client — and it
		// starts from the ACCUMULATED history plus the recovery turns, so
		// the search the loop already served is never lost.
		if kind, doRetry := s.retryKind(res, calls); doRetry {
			if retryRes, retryCalls, _, err2 := s.resolveNative(r.Context(), ureq, plan, nativeRetryHistory(hist, res, kind), rtag); err2 == nil {
				res, calls = retryRes, retryCalls
			} else {
				s.upErrors.Add(1)
				s.logf("tools: native recovery retry failed: %v (passing original round through)", err2)
			}
		}
		s.writeToolCompletion(w, model, res.content, res.reasoning, calls, res.annotations())
		return
	}
	if kind, retry := s.retryKind(res, nil); retry {
		res = s.recoverRound(r.Context(), ureq, manifest, res, kind)
	}
	s.writeToolCompletion(w, model, res.content, res.reasoning, res.calls, res.annotations())
}

// writeToolCompletion writes the aggregated chat.completion for tool output.
func (s *Server) writeToolCompletion(w http.ResponseWriter, model, content, reasoning string, calls []tools.Call, cites []openai.Citation) {
	msg := openai.CompletionMessage{
		Role:             "assistant",
		Content:          content,
		ReasoningContent: reasoning,
		Annotations:      cites,
	}
	finish := "stop"
	if len(calls) > 0 {
		msg.ToolCalls = make([]openai.ToolCall, 0, len(calls))
		for _, c := range calls {
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
				ID:       tools.NewID(),
				Type:     "function",
				Function: openai.FunctionCall{Name: c.Name, Arguments: c.Arguments},
			})
		}
		finish = "tool_calls"
	}
	writeJSON(w, http.StatusOK, openai.Completion{
		ID:      upstream.RandomID("chatcmpl"),
		Object:  "chat.completion",
		Created: s.now().Unix(),
		Model:   model,
		Choices: []openai.CompletionChoice{{
			Index:        0,
			Message:      msg,
			FinishReason: finish,
		}},
	})
}
