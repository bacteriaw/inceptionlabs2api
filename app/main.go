// Command inceptionlabs2api exposes chat.inceptionlabs.ai (Mercury 3) as an
// OpenAI-compatible Chat Completions API.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	"inceptionlabs2api/internal/config"
	"inceptionlabs2api/internal/pool"
	"inceptionlabs2api/internal/server"
	"inceptionlabs2api/internal/upstream"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

// run builds the gateway and serves until SIGINT/SIGTERM. It returns an error
// instead of exiting so the wiring is testable.
func run() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	logf := log.Printf

	up, fetch := upstream.NewClient(upstream.Config{
		BaseURL:           cfg.UpstreamBase,
		Proxy:             cfg.Proxy,
		SessionTimeout:    cfg.SessionTimeout,
		SessionPerRequest: cfg.SessionPerRequest,
		ClearanceRefresh:  cfg.ClearanceRefresh,
		ClearanceCooldown: cfg.ClearanceCooldown,
		ClearancePool:     cfg.ClearancePool,
		Logf:              logf,
	})

	// 每请求新 session 时池子不再预热（Min=0），避免白白多取 token。
	poolMin := cfg.PoolMin
	if cfg.SessionPerRequest {
		poolMin = 0
	}
	p := pool.New(pool.Config{
		Min:      poolMin,
		Max:      cfg.PoolMax,
		TTL:      cfg.PoolTTL,
		Interval: 15 * time.Second,
		OnFetchError: func(err error) {
			logf("pool: refill failed (serving cached tokens if any): %v", err)
		},
	}, fetch)
	defer p.Close()
	up.BindPool(p)

	// Fill the clearance ring's first slot at boot (~0.1-0.4 s, so the first
	// chat request does not pay the PoW + round trip) and start the one
	// background loop that fills the rest, refills retired slots and refreshes
	// stale ones (SPEC §2.1.1, §2.1.2). A boot failure is logged, never fatal:
	// the service starts and the loop retries. The pool then fills behind it
	// (it needs clearance to fetch a token), and the search probe runs last
	// (it needs a pooled token).
	bootCtx, stopBoot := context.WithCancel(context.Background())
	defer stopBoot()
	up.StartClearancePrewarm(bootCtx)

	srv := server.New(server.Config{
		APIKey:             cfg.APIKey,
		Model:              cfg.Model,
		Models:             cfg.Models,
		IdleTimeout:        cfg.IdleTimeout,
		MaxMessages:        cfg.MaxMessages,
		EmptyRetries:       cfg.EmptyRetries,
		CompactKeepRecent:  cfg.CompactKeepRecent,
		SearchTimeout:      cfg.SearchTimeout,
		SearchProbeTimeout: cfg.SearchProbeTimeout,
		AudioTimeout:       cfg.AudioTimeout,
		AudioMaxBytes:      cfg.AudioMaxBytes,
		DecisionTimeout:    cfg.DecisionTimeout,
		PreambleStyle:      cfg.PreambleStyle,
		NativeWebSearch:    cfg.NativeWebSearch,
		Logf:               logf,
	}, up, p)
	srv.SetSearcher(up)
	srv.SetSpeech(up)
	srv.SetDecider(up)
	srv.SetClearance(up.Clearance())
	if cfg.SearchConfigCheck {
		// Background and bounded: a challenged config route must never delay
		// or fail boot.
		srv.StartSearchProbe()
	}

	httpSrv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		logf("inceptionlabs2api listening on %s (upstream=%s proxy=%q models=%v session_per_request=%v)",
			cfg.Addr, cfg.UpstreamBase, config.RedactProxy(cfg.Proxy), cfg.Models, cfg.SessionPerRequest)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("http server: %v", err)
		}
	}()

	<-ctx.Done()
	logf("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}
	return nil
}
