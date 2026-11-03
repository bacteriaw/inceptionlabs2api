// Package pool keeps a small set of anonymous upstream session tokens.
//
// Tokens are free and not bound to an IP, so the pool is deliberately simple:
// it holds between Min and Max tokens, replaces them on TTL expiry, and
// discards a token only when the upstream explicitly rejects it (401/403).
// There is no error counting and no in-flight accounting.
package pool

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Token shape bounds: {unix_ts}.{32 hex}.{64 hex}, where the unix timestamp
// is 10 digits (11 after 2286).
const (
	// refillTimeout bounds the asynchronous refill started by Discard.
	refillTimeout = 15 * time.Second

	tokenParts  = 3
	nonceHexLen = 32
	signHexLen  = 64
	minTokenLen = 10 + 1 + nonceHexLen + 1 + signHexLen
	maxTokenLen = 11 + 1 + nonceHexLen + 1 + signHexLen
)

// ErrNoToken is returned by Get when no live token exists and a fetch is not
// possible or fails.
var ErrNoToken = errors.New("pool: no session token available")

// Payload is the upstream /api/session response.
type Payload struct {
	OK    bool   `json:"ok"`
	Token string `json:"token"`
}

// Fetcher obtains one fresh token from the upstream session endpoint.
type Fetcher func(ctx context.Context) (string, error)

// Token is a pooled session token with its acquisition time.
type Token struct {
	Value   string
	Created time.Time
}

// Config controls pool sizing and lifetime.
type Config struct {
	Min      int
	Max      int
	TTL      time.Duration
	Interval time.Duration
	// Now and NewTicker are injectable for tests. Nil means real time.
	Now       func() time.Time
	NewTicker func(time.Duration) *time.Ticker
	// OnFetchError is called for background refill failures. Nil means the
	// error is logged.
	OnFetchError func(error)
}

// Pool is a concurrency-safe set of anonymous session tokens.
type Pool struct {
	fetch Fetcher
	cfg   Config
	logf  func(format string, args ...any)

	mu     sync.Mutex
	tokens []Token

	// fetchMu serialises synchronous and background fetches so a burst of
	// requests cannot start a handshake storm.
	fetchMu sync.Mutex

	// refilling guards Discard's async refill: true while one refill
	// goroutine is live. Storms of discards collapse into one refill.
	refilling bool

	stop     chan struct{}
	stopOnce sync.Once
	wg       sync.WaitGroup
}

// New starts a pool and launches its maintainer. It never fails because the
// upstream is unreachable: an empty pool serves 503 until a token arrives.
func New(cfg Config, fetch Fetcher) *Pool {
	if cfg.Min < 0 {
		cfg.Min = 0
	}
	if cfg.Max < 1 {
		cfg.Max = 1
	}
	if cfg.Min > cfg.Max {
		cfg.Min = cfg.Max
	}
	if cfg.TTL <= 0 {
		cfg.TTL = 30 * time.Minute
	}
	if cfg.Interval <= 0 {
		cfg.Interval = 15 * time.Second
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.NewTicker == nil {
		cfg.NewTicker = time.NewTicker
	}
	p := &Pool{
		fetch: fetch,
		cfg:   cfg,
		logf:  log.Printf,
		stop:  make(chan struct{}),
	}
	if cfg.OnFetchError != nil {
		p.logf = func(string, ...any) {}
	}
	p.wg.Add(1)
	go p.maintain()
	return p
}

// Close stops the maintainer and waits for it to exit.
func (p *Pool) Close() {
	p.stopOnce.Do(func() { close(p.stop) })
	p.wg.Wait()
}

// Stats is a snapshot of pool state, exposed by /healthz.
type Stats struct {
	Size    int `json:"size"`
	Min     int `json:"min"`
	Max     int `json:"max"`
	Expired int `json:"expired"`
}

// Stats returns the current pool size and configured bounds.
func (p *Pool) Stats() Stats {
	p.mu.Lock()
	defer p.mu.Unlock()
	now := p.cfg.Now()
	var live, expired int
	for _, t := range p.tokens {
		if now.Sub(t.Created) >= p.cfg.TTL {
			expired++
			continue
		}
		live++
	}
	return Stats{Size: live, Min: p.cfg.Min, Max: p.cfg.Max, Expired: expired}
}

// Get returns a live token, round-robin. If none is available it fetches one
// synchronously, bounded by ctx. The returned error wraps ErrNoToken when the
// fetch fails.
func (p *Pool) Get(ctx context.Context) (string, error) {
	if tok, ok := p.pick(); ok {
		return tok, nil
	}
	if err := p.ensure(ctx); err != nil {
		return "", fmt.Errorf("%w: %w", ErrNoToken, err)
	}
	if tok, ok := p.pick(); ok {
		return tok, nil
	}
	return "", ErrNoToken
}

// Discard removes exactly tok and triggers one asynchronous refill. It is
// called only when the upstream answered 401/403 for that token. Refills are
// single-flight: a burst of discards (429/401 storm) must not stack one
// refill goroutine per token — grow() is idempotent, so one live refill
// covers them all.
func (p *Pool) Discard(tok string) {
	p.mu.Lock()
	removed := false
	for i := range p.tokens {
		if p.tokens[i].Value == tok {
			p.tokens = append(p.tokens[:i], p.tokens[i+1:]...)
			removed = true
			break
		}
	}
	storm := removed && p.refilling
	if removed && !p.refilling {
		p.refilling = true
	}
	p.mu.Unlock()
	if !removed {
		return
	}
	if storm {
		// A refill is already running; grow() tops up to Min on its next
		// pass, so stacking another goroutine only burns handshakes.
		return
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), refillTimeout)
		defer cancel()
		if err := p.grow(ctx); err != nil {
			p.reportFetchError(err)
		}
		p.mu.Lock()
		p.refilling = false
		p.mu.Unlock()
	}()
}

// pick returns a live token using a round-robin cursor derived from the token
// slice order.
func (p *Pool) pick() (string, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.dropExpiredLocked()
	if len(p.tokens) == 0 {
		return "", false
	}
	t := p.tokens[0]
	p.tokens = append(p.tokens[1:], t)
	return t.Value, true
}

// ensure fills the pool to Min, and always fetches at least one token when it
// is empty. That keeps Min=0 a valid lazy mode instead of a permanently empty
// pool that answers every request with 503.
func (p *Pool) ensure(ctx context.Context) error {
	p.fetchMu.Lock()
	defer p.fetchMu.Unlock()

	for {
		p.mu.Lock()
		p.dropExpiredLocked()
		need := p.cfg.Min - len(p.tokens)
		if len(p.tokens) == 0 && need < 1 {
			need = 1
		}
		full := len(p.tokens) >= p.cfg.Max
		p.mu.Unlock()
		if need <= 0 || full {
			return nil
		}
		tok, err := p.fetch(ctx)
		if err != nil {
			return err
		}
		created := p.cfg.Now()
		p.mu.Lock()
		if len(p.tokens) < p.cfg.Max {
			p.tokens = append(p.tokens, Token{Value: tok, Created: created})
		}
		p.mu.Unlock()
	}
}

// grow fetches tokens until Min is reached or Max is hit.
func (p *Pool) grow(ctx context.Context) error {
	p.fetchMu.Lock()
	defer p.fetchMu.Unlock()
	for {
		p.mu.Lock()
		p.dropExpiredLocked()
		need := p.cfg.Min - len(p.tokens)
		full := len(p.tokens) >= p.cfg.Max
		p.mu.Unlock()
		if need <= 0 || full {
			return nil
		}
		tok, err := p.fetch(ctx)
		if err != nil {
			return err
		}
		created := p.cfg.Now()
		p.mu.Lock()
		if len(p.tokens) < p.cfg.Max {
			p.tokens = append(p.tokens, Token{Value: tok, Created: created})
		}
		p.mu.Unlock()
	}
}

func (p *Pool) dropExpiredLocked() {
	now := p.cfg.Now()
	kept := p.tokens[:0]
	for _, t := range p.tokens {
		if now.Sub(t.Created) < p.cfg.TTL {
			kept = append(kept, t)
		}
	}
	p.tokens = kept
}

func (p *Pool) maintain() {
	defer p.wg.Done()
	ticker := p.cfg.NewTicker(p.cfg.Interval)
	defer ticker.Stop()
	for {
		select {
		case <-p.stop:
			return
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), p.cfg.Interval/2)
			err := p.grow(ctx)
			cancel()
			if err != nil {
				p.reportFetchError(err)
			}
		}
	}
}

func (p *Pool) reportFetchError(err error) {
	if p.cfg.OnFetchError != nil {
		p.cfg.OnFetchError(err)
		return
	}
	p.logf("pool: refill: %v", err)
}

// ValidateToken checks the {unix_ts}.{32 hex}.{64 hex} shape.
func ValidateToken(tok string) bool {
	if len(tok) < minTokenLen || len(tok) > maxTokenLen {
		return false
	}
	parts := strings.Split(tok, ".")
	if len(parts) != tokenParts {
		return false
	}
	if _, err := strconv.ParseInt(parts[0], 10, 64); err != nil {
		return false
	}
	if len(parts[1]) != nonceHexLen || !isHex(parts[1]) {
		return false
	}
	if len(parts[2]) != signHexLen || !isHex(parts[2]) {
		return false
	}
	return true
}

func isHex(s string) bool {
	for _, c := range s {
		switch {
		case c >= '0' && c <= '9':
		case c >= 'a' && c <= 'f':
		case c >= 'A' && c <= 'F':
		default:
			return false
		}
	}
	return true
}
