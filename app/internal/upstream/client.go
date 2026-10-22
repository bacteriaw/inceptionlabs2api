package upstream

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"inceptionlabs2api/internal/clear"
	"inceptionlabs2api/internal/pool"
)

// maxErrorBody caps how much of a non-200 upstream body is read (and logged).
const maxErrorBody = 4096

// maxClientMessage bounds the upstream text echoed to clients.
const maxClientMessage = 1000

// maxAuthRetries is the number of additional attempts after the first when
// the upstream rejects a session token with 401/403.
const maxAuthRetries = 2

// StatusError is a typed non-200 upstream response. The body is kept for
// server-side logging and for a sanitised client-facing message.
type StatusError struct {
	Status int
	Body   string
}

// Error implements error.
func (e *StatusError) Error() string {
	return fmt.Sprintf("upstream: status %d: %s", e.Status, truncate(e.Body, 200))
}

// Message returns the upstream's own error text for a client-facing envelope.
// The upstream speaks for itself; the gateway only strips HTML (a challenge
// page is not an error message) and bounds the length. It returns "" when
// there is nothing worth repeating.
func (e *StatusError) Message() string {
	body := strings.TrimSpace(e.Body)
	if body == "" || strings.HasPrefix(body, "<") || strings.Contains(body, "<html") {
		return ""
	}
	// Some upstream errors are JSON envelopes; surface their message field so
	// the client sees prose rather than raw JSON inside a JSON string. The
	// audio routes answer `{"error":"..."}` (a bare string) while chat uses
	// the nested object form, so both are accepted.
	var env struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
	}
	if json.Unmarshal([]byte(body), &env) == nil {
		var nested struct {
			Message string `json:"message"`
		}
		if json.Unmarshal(env.Error, &nested) == nil {
			if msg := strings.TrimSpace(nested.Message); msg != "" {
				return truncate(msg, maxClientMessage)
			}
		}
		var bare string
		if json.Unmarshal(env.Error, &bare) == nil {
			if msg := strings.TrimSpace(bare); msg != "" {
				return truncate(msg, maxClientMessage)
			}
		}
		if msg := strings.TrimSpace(env.Message); msg != "" {
			return truncate(msg, maxClientMessage)
		}
	}
	return truncate(body, maxClientMessage)
}

// ErrUnavailable means no session token could be obtained.
var ErrUnavailable = errors.New("upstream: no session token available")

// ErrAuthRejected means the upstream answered 401/403 on every attempt.
var ErrAuthRejected = errors.New("upstream: session token rejected")

// ErrClearance means the checkpoint could not be cleared.
var ErrClearance = errors.New("upstream: clearance unavailable")

// Client issues upstream requests through one shared HTTP/2 transport and the
// clearance-aware request path.
type Client struct {
	base    string
	pool    *pool.Pool
	timeout time.Duration
	// sessionPerRequest 为真时 authed 每次请求都新取 session，跳过池复用。
	sessionPerRequest bool
	logf    func(format string, args ...any)
	clear   *clear.Clearance
	hc      *http.Client
}

// Config configures a Client.
type Config struct {
	BaseURL string
	Proxy   string
	Pool    *pool.Pool
	// SessionTimeout bounds one /api/session fetch.
	SessionTimeout time.Duration
	// SessionPerRequest fetches a brand-new anonymous session token for every
	// upstream call instead of drawing from the pool. The upstream keeps the
	// conversation history on that token, so reuse eventually fails with
	// "This conversation is too long".
	SessionPerRequest bool
	// ClearanceRefresh is the proactive cookie refresh age. Zero means 45m.
	ClearanceRefresh time.Duration
	// ClearanceCooldown is the minimum retry interval after a failed
	// acquisition. Zero means 15s.
	ClearanceCooldown time.Duration
	// ClearancePool is the size of the clearance ring. Zero means 20.
	ClearancePool int
	// ClearanceFillPace is the pause the background fill loop takes between
	// acquisitions. Zero means 1.5s; negative disables it (tests).
	ClearanceFillPace time.Duration
	// ClearanceJitter is the extra random pause added to ClearanceFillPace.
	// Zero means 500ms; negative disables it (tests).
	ClearanceJitter time.Duration
	// RootCAs is optional; nil means the system trust store.
	RootCAs *x509.CertPool
	// Logf is optional; nil means log.Printf.
	Logf func(format string, args ...any)
}

// NewClient builds a Client and returns it together with a Fetcher suitable
// for pool.New. The transport is created once and shared by session fetches,
// chat, search, and clearance acquisition, so every request egresses through
// the same proxy with the same TLS configuration.
func NewClient(cfg Config) (*Client, pool.Fetcher) {
	if cfg.SessionTimeout <= 0 {
		cfg.SessionTimeout = 15 * time.Second
	}
	logf := cfg.Logf
	if logf == nil {
		logf = log.Printf
	}
	tr := clear.Transport(clear.TransportOptions{Proxy: cfg.Proxy, RootCAs: cfg.RootCAs})
	// No http.Client.Timeout: a wall-clock cap would kill legitimately long
	// diffusion generations. The sliding idle watchdog lives in Reader.
	hc := &http.Client{Transport: tr, CheckRedirect: checkRedirect}
	c := &Client{
		base:              strings.TrimRight(cfg.BaseURL, "/"),
		pool:              cfg.Pool,
		timeout:           cfg.SessionTimeout,
		sessionPerRequest: cfg.SessionPerRequest,
		logf:              logf,
		hc:                hc,
	}
	c.clear = clear.New(clear.Config{
		BaseURL:  c.base,
		Client:   hc,
		Pool:     cfg.ClearancePool,
		Refresh:  cfg.ClearanceRefresh,
		Cooldown: cfg.ClearanceCooldown,
		FillPace: cfg.ClearanceFillPace,
		Jitter:   cfg.ClearanceJitter,
		Logf:     logf,
	})
	return c, c.FetchSession
}

// BindPool attaches the token pool the client draws from. It is called once
// at startup because the pool itself depends on the client's Fetcher.
func (c *Client) BindPool(p *pool.Pool) { c.pool = p }

// Clearance exposes the clearance state for /healthz.
func (c *Client) Clearance() *clear.Clearance { return c.clear }

// StartClearancePrewarm fills the first clearance slot and then starts the one
// background fill loop (SPEC §2.1.1, §2.1.2):
//
//   - Boot waits for the first slot synchronously: ~0.1–0.4 s, "warm a base,
//     not the maximum". The server starts as soon as this returns.
//   - A failed boot fill is NOT a failed startup: it is logged, the service
//     starts with `has_cookie:false`, and the loop keeps retrying.
//   - The loop fills the remaining slots sequentially (1–2 s + jitter apart),
//     refills any slot retired by a 429, and replaces stale slots one per
//     iteration.
//
// The loop stops when ctx is cancelled.
func (c *Client) StartClearancePrewarm(ctx context.Context) {
	if err := c.clear.PrewarmBoot(ctx); err != nil {
		c.logf("clear: boot prewarm failed (the service still starts; the fill loop retries): %v", err)
	}
	c.clear.StartFillLoop(ctx)
}

// applyClearance sets the shared UA and the clearance cookie on req.
func (c *Client) applyClearance(req *http.Request) {
	ua, cookie := c.clear.Header()
	req.Header.Set("User-Agent", ua)
	if cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
}

// challengedCookie extracts the clearance cookie value from the request that
// produced a challenged response, so only that ring slot is retired. It is ""
// when the request carried none (an unprotected probe), which Retire ignores.
func challengedCookie(resp *http.Response) string {
	if resp == nil || resp.Request == nil {
		return ""
	}
	for _, ck := range resp.Request.Cookies() {
		if ck.Name == clear.CookieName {
			return ck.Value
		}
	}
	return ""
}

// retireClearance retires the ring slot a challenge just rejected. It never
// solves: the fill loop notices the empty slot and refills it (SPEC §2.1.2).
// Only that slot is touched, so requests rotating over the other slots keep
// their cookies and ages.
func (c *Client) retireClearance(cookie string) {
	if err := c.clear.Retire(cookie); err != nil {
		c.logf("upstream: retiring clearance slot: %v", err)
	}
}

// ensureClearance acquires (or refreshes) the clearance cookie. It is the
// request path's lazy fallback for a ring with nothing live; it never tops up
// a partially filled ring (the fill loop does that).
func (c *Client) ensureClearance(ctx context.Context) error {
	if err := c.clear.Ensure(ctx); err != nil {
		return fmt.Errorf("%w: %v", ErrClearance, err)
	}
	return nil
}

// Send performs an upstream request with clearance handling. prepare
// customises headers. On a 429 that carries a Vercel challenge, clearance is
// refreshed once and the request (with a fresh body reader, if any) is
// replayed. The caller owns the returned response body.
//
// This is the one request path: session fetches, chat, search, and probes all
// funnel through it, so no endpoint can accidentally bypass the cookie.
func (c *Client) Send(ctx context.Context, method, path string, body []byte, headerTimeout time.Duration, prepare func(*http.Request)) (*http.Response, error) {
	if err := c.ensureClearance(ctx); err != nil {
		return nil, err
	}
	resp, err := c.do(ctx, method, path, body, headerTimeout, prepare)
	if err != nil {
		return nil, err
	}
	if !clear.IsChallenge(resp) {
		return resp, nil
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxErrorBody))
	resp.Body.Close()
	c.logf("upstream: checkpoint challenged %s %s; retiring the slot and replaying once", method, path)

	// The challenge proves *that* cookie is not accepted. Retire only the slot
	// that carried it and replay immediately on the next live slot: the request
	// path never waits for a solve. The fill loop refills the retired slot
	// (SPEC §2.1.2). With a ring of one, the lazy Ensure below performs the
	// solve because there is no other slot to replay on.
	c.retireClearance(challengedCookie(resp))
	if err := c.ensureClearance(ctx); err != nil {
		return nil, err
	}
	resp, err = c.do(ctx, method, path, body, headerTimeout, prepare)
	if err != nil {
		return nil, err
	}
	if clear.IsChallenge(resp) {
		// Replaying once already failed. Returning the challenge body would
		// let a caller misread it as an ordinary response (e.g. an empty
		// capability probe); say what happened instead.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxErrorBody))
		resp.Body.Close()
		return nil, fmt.Errorf("%w: the checkpoint challenged %s even after a fresh clearance", ErrClearance, path)
	}
	return resp, nil
}

// do issues one upstream request. A nil body means no request body.
// headerTimeout bounds the whole pre-response phase (dial, TLS, headers); the
// response body is not subject to it, so long generations are unaffected.
func (c *Client) do(ctx context.Context, method, path string, body []byte, headerTimeout time.Duration, prepare func(*http.Request)) (*http.Response, error) {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	if headerTimeout <= 0 {
		headerTimeout = 60 * time.Second
	}
	hctx, cancelHeader := context.WithCancel(ctx)
	timer := time.AfterFunc(headerTimeout, cancelHeader)

	req, err := http.NewRequestWithContext(hctx, method, c.base+path, rdr)
	if err != nil {
		timer.Stop()
		cancelHeader()
		return nil, fmt.Errorf("upstream: build %s request: %w", path, err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	c.applyClearance(req)
	if prepare != nil {
		prepare(req)
	}
	resp, err := c.hc.Do(req)
	timer.Stop()
	if err != nil {
		timedOut := hctx.Err() != nil && ctx.Err() == nil
		cancelHeader()
		if timedOut {
			return nil, fmt.Errorf("%w: no response headers within %s", ErrIdleTimeout, headerTimeout)
		}
		return nil, fmt.Errorf("upstream: %s request: %w", path, err)
	}
	resp.Body = &closeHook{ReadCloser: resp.Body, hook: cancelHeader}
	return resp, nil
}

// closeHook runs hook once when the body is closed, releasing the per-request
// header timer context.
type closeHook struct {
	io.ReadCloser
	hook func()
}

func (c *closeHook) Close() error {
	err := c.ReadCloser.Close()
	c.hook()
	return err
}

// FetchSession calls GET /api/session once and validates the token shape.
func (c *Client) FetchSession(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	resp, err := c.Send(ctx, http.MethodGet, "/api/session", nil, c.timeout, func(req *http.Request) {
		req.Header.Set("Accept", "*/*")
	})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("upstream: read session response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", &StatusError{Status: resp.StatusCode, Body: string(body)}
	}
	var p pool.Payload
	if err := json.Unmarshal(body, &p); err != nil {
		return "", fmt.Errorf("upstream: parse session response: %w (body %q)", err, truncate(string(body), 200))
	}
	if !p.OK || p.Token == "" {
		return "", fmt.Errorf("upstream: session response ok=%v token present=%v", p.OK, p.Token != "")
	}
	if !pool.ValidateToken(p.Token) {
		return "", fmt.Errorf("upstream: malformed session token %q", truncate(p.Token, 16))
	}
	return p.Token, nil
}

// checkRedirect rejects redirects that leave the configured origin or
// downgrade HTTPS. The session token travels as a custom header, which Go
// would otherwise copy to a cross-host redirect target. The clearance cookie
// is scoped to the origin for the same reason.
func checkRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 3 {
		return fmt.Errorf("upstream: stopped after %d redirects", len(via))
	}
	orig := via[0].URL
	if req.URL.Host != orig.Host {
		return fmt.Errorf("upstream: refusing cross-host redirect to %s", req.URL.Host)
	}
	if orig.Scheme == "https" && req.URL.Scheme != "https" {
		return fmt.Errorf("upstream: refusing HTTPS downgrade on redirect")
	}
	return nil
}

// Stream is an opened upstream SSE stream. Close releases the response body.
type Stream struct {
	Body   io.ReadCloser
	Reader *Reader
}

// Close closes the response body.
func (s *Stream) Close() error { return s.Body.Close() }

// Chat opens a stream for req. It acquires a token, POSTs, and retries with a
// fresh token on 401/403, at most maxAuthRetries times. It returns a typed
// *StatusError for other non-200 responses and an error wrapping
// ErrUnavailable when the pool cannot supply a token. Client cancellation and
// transport failures never discard a token.
func (c *Client) Chat(ctx context.Context, req *ChatRequest, idle time.Duration, onMalformed func(string)) (*Stream, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("upstream: marshal chat request: %w", err)
	}
	resp, err := c.postAuthed(ctx, "/api/chat", "chat", body, idle, func(req *http.Request) {
		req.Header.Set("Accept", "text/event-stream, application/json")
	})
	if err != nil {
		return nil, err
	}
	return &Stream{
		Body:   resp.Body,
		Reader: NewReader(resp.Body, idle, onMalformed),
	}, nil
}

// sessionToken returns the token to use for one request: a freshly fetched
// session when SessionPerRequest is on, otherwise a pooled one.
func (c *Client) sessionToken(ctx context.Context) (string, error) {
	if c.sessionPerRequest {
		return c.FetchSession(ctx)
	}
	return c.pool.Get(ctx)
}

// authed issues one request with a pooled token, retrying with a fresh token
// on 401/403 at most maxAuthRetries times. It returns a 200 response whose
// body the caller owns. A 401/403 discards that token; other non-200 statuses
// are a typed *StatusError and keep the token, since the pool never poisons a
// token for an upstream fault. prepare may customise headers.
//
// This is the one token-bearing request path (§6.1): chat, search, TTS, STT
// and the search-config probe all call it, so none of them can grow its own
// retry loop or forget to discard a rejected token.
func (c *Client) authed(ctx context.Context, method, path, label string, body []byte, headerTimeout time.Duration, prepare func(*http.Request)) (*http.Response, error) {
	if c.pool == nil {
		return nil, fmt.Errorf("%w: no pool bound", ErrUnavailable)
	}
	var lastErr error
	rateRetries := 0
	for attempt := 0; attempt <= maxAuthRetries; attempt++ {
		token, err := c.sessionToken(ctx)
		if err != nil {
			// A pool that could not fill because the checkpoint refused us is
			// a clearance failure, not a generic "no token": the operator can
			// act on it (egress, UA, PoW drift).
			if errors.Is(err, ErrClearance) {
				return nil, fmt.Errorf("%w: %w", ErrClearance, err)
			}
			return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
		}
		resp, err := c.Send(ctx, method, path, body, headerTimeout, func(req *http.Request) {
			req.Header.Set("x-session-token", token)
			if prepare != nil {
				prepare(req)
			}
		})
		if err != nil {
			return nil, err
		}
		switch {
		case resp.StatusCode == http.StatusOK:
			return resp, nil
		case resp.StatusCode == http.StatusTooManyRequests && rateRetries < 2:
			// [local patch] 每请求新 session 会抬高上游调用速率，突发时上游回
			// 429 "Too Many Requests"（无 Vercel challenge 头）。退避后重试，
			// 不消耗 401/403 的重试预算。
			bodyText, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
			resp.Body.Close()
			rateRetries++
			delay := time.Duration(rateRetries) * 1500 * time.Millisecond
			c.logf("upstream: %s 429 too many requests, backing off %s then retrying", label, delay)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			lastErr = &StatusError{Status: resp.StatusCode, Body: string(bodyText)}
			attempt-- // 不占用鉴权重试次数
			continue
		case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxErrorBody))
			resp.Body.Close()
			if !c.sessionPerRequest {
				c.pool.Discard(token)
			}
			c.logf("upstream: session token rejected (status %d), retrying", resp.StatusCode)
			lastErr = fmt.Errorf("%w: status %d", ErrAuthRejected, resp.StatusCode)
			continue
		default:
			bodyText, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
			resp.Body.Close()
			c.logf("upstream: %s status %d: %s", label, resp.StatusCode, truncate(string(bodyText), 500))
			return nil, &StatusError{Status: resp.StatusCode, Body: string(bodyText)}
		}
	}
	return nil, lastErr
}

// postAuthed is the POST form of authed, used by every JSON/audio family.
func (c *Client) postAuthed(ctx context.Context, path, label string, body []byte, headerTimeout time.Duration, prepare func(*http.Request)) (*http.Response, error) {
	return c.authed(ctx, http.MethodPost, path, label, body, headerTimeout, prepare)
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// collapseSpace collapses runs of whitespace (and newlines/tabs) to single
// spaces and trims the ends. The upstream titles arrive with raw HTML
// whitespace; both the search results and the citation annotations use this
// one rule (SPEC §6.1).
func collapseSpace(s string) string { return strings.Join(strings.Fields(s), " ") }

// clampInt returns v bounded to [lo,hi], or def when v is unset (zero). It is
// the translation a family applies to a numeric field whose range the upstream
// publishes; the server's search mapping uses it for fanout/depth (§7.1).
func ClampInt(v, lo, hi, def int) int {
	if v == 0 {
		return def
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
