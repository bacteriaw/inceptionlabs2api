package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"inceptionlabs2api/internal/openai"
	"inceptionlabs2api/internal/upstream"
)

// stream writes the OpenAI SSE response. Headers are committed only after
// the upstream stream is open, so pool and upstream failures still carry a
// proper HTTP status.
func (s *Server) stream(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string) {
	rtag := wireTag()
	wireLog(rtag, "PLAIN STREAM (no tools path)")

	// The upstream sometimes ends a stream with no text at all (measured
	// 2026-10-22: mercury-3, 80k-130k token prompts, completion_tokens=0).
	// A silent empty answer is indistinguishable from a broken turn downstream,
	// so retry it here and only commit headers once real text arrives.
	attempts := s.emptyAttempts()
	for attempt := 1; attempt <= attempts; attempt++ {
		stream, err := s.up.Chat(r.Context(), ureq, s.cfg.IdleTimeout, func(line string) {
			s.logf("upstream: skipping malformed SSE line: %q", truncate(line, 200))
		})
		if err != nil {
			s.fail(w, s.upstreamError(err))
			return
		}
		s.streams.Add(1)

		id := upstream.RandomID("chatcmpl")
		created := s.now().Unix()
		wrote := false
		flusher, _ := w.(http.Flusher)
		write := func(delta openai.Delta, finish *string) bool {
			if !wrote {
				setStreamHeaders(w)
				wrote = true
			}
			chunk := openai.Chunk{
				ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
				Choices: []openai.ChunkChoice{{Index: 0, Delta: delta, FinishReason: finish}},
			}
			if err := writeEvent(w, chunk); err != nil {
				return false
			}
			if flusher != nil {
				flusher.Flush()
			}
			return true
		}

		// Deltas are buffered until the first visible text so an empty answer
		// can still be retried with a proper HTTP status.
		var pending []openai.Delta
		var cites upstream.CitationCollector
		visible := false
		readErr := stream.Reader.Read(r.Context(), func(ev upstream.Event) error {
			cites.Observe(ev, s.logf)
			defer wireDumpEvent(rtag, ev, 0, 0)
			switch ev.Type {
			case upstream.EventReasoningDelta:
				text := ev.Reasoning()
				if text == "" {
					return nil
				}
				if visible {
					if !write(openai.Delta{ReasoningContent: text}, nil) {
						return errClientGone
					}
				} else {
					pending = append(pending, openai.Delta{ReasoningContent: text})
				}
			case upstream.EventTextStart, upstream.EventTextDelta:
				text := ev.ContentText()
				if text == "" {
					return nil
				}
				if !visible {
					visible = true
					if !write(openai.Delta{Role: "assistant"}, nil) {
						return errClientGone
					}
					for _, d := range pending {
						if !write(d, nil) {
							return errClientGone
						}
					}
					pending = nil
				}
				if !write(openai.Delta{Content: text}, nil) {
					return errClientGone
				}
			}
			return nil
		})
		stream.Close()

		if readErr != nil {
			if errors.Is(readErr, errClientGone) {
				return
			}
			if errors.Is(readErr, context.Canceled) && r.Context().Err() != nil {
				return
			}
			if wrote {
				s.upErrors.Add(1)
				s.logf("upstream: stream failed after headers: %v", readErr)
				se := streamError(readErr)
				_ = writeEvent(w, se.Envelope())
				_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
				if flusher != nil {
					flusher.Flush()
				}
				return
			}
			s.fail(w, s.upstreamError(readErr))
			return
		}

		if !visible {
			s.logf("upstream: empty completion on attempt %d/%d (no text; %d buffered reasoning delta(s))", attempt, attempts, len(pending))
			continue
		}

		stop := "stop"
		if !write(openai.Delta{Annotations: cites.Citations()}, &stop) {
			return
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	// Every attempt came back empty: report a retryable upstream error instead
	// of a silent 200 with no content.
	s.fail(w, emptyUpstreamError())
}

// errClientGone aborts the read loop when writing to the client fails.
var errClientGone = errors.New("server: client write failed")

// complete aggregates the upstream stream into a single chat.completion.
func (s *Server) complete(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string) {
	attempts := s.emptyAttempts()
	for attempt := 1; attempt <= attempts; attempt++ {
		completion, empty, err := s.readCompletion(r.Context(), ureq, model)
		if err != nil {
			if errors.Is(err, context.Canceled) && r.Context().Err() != nil {
				return
			}
			s.fail(w, s.upstreamError(err))
			return
		}
		if !empty {
			writeJSON(w, http.StatusOK, completion)
			return
		}
		s.logf("upstream: empty completion on attempt %d/%d (no text)", attempt, attempts)
	}
	s.fail(w, emptyUpstreamError())
}

// readCompletion aggregates one upstream stream into a chat.completion and
// reports whether the answer carried no visible text at all.
func (s *Server) readCompletion(ctx context.Context, ureq *upstream.ChatRequest, model string) (openai.Completion, bool, error) {
	stream, err := s.up.Chat(ctx, ureq, s.cfg.IdleTimeout, func(line string) {
		s.logf("upstream: skipping malformed SSE line: %q", truncate(line, 200))
	})
	if err != nil {
		return openai.Completion{}, false, err
	}
	defer stream.Close()

	var content, reasoning strings.Builder
	var cites upstream.CitationCollector
	readErr := stream.Reader.Read(ctx, func(ev upstream.Event) error {
		cites.Observe(ev, s.logf)
		switch ev.Type {
		case upstream.EventReasoningDelta:
			reasoning.WriteString(ev.Reasoning())
		case upstream.EventTextStart, upstream.EventTextDelta:
			content.WriteString(ev.ContentText())
		}
		return nil
	})
	if readErr != nil {
		return openai.Completion{}, false, readErr
	}

	completion := openai.Completion{
		ID:      upstream.RandomID("chatcmpl"),
		Object:  "chat.completion",
		Created: s.now().Unix(),
		Model:   model,
		Choices: []openai.CompletionChoice{{
			Index: 0,
			Message: openai.CompletionMessage{
				Role:             "assistant",
				Content:          content.String(),
				ReasoningContent: reasoning.String(),
				Annotations:      cites.Citations(),
			},
			FinishReason: "stop",
		}},
	}
	return completion, content.Len() == 0, nil
}

// emptyAttempts is the total number of upstream tries for one client request
// (1 initial + EmptyRetries).
func (s *Server) emptyAttempts() int {
	n := s.cfg.EmptyRetries + 1
	if n < 1 {
		n = 1
	}
	return n
}

// emptyUpstreamError is the client-facing error for an upstream that keeps
// answering with no text. 502 lets OpenAI-compatible clients retry.
func emptyUpstreamError() *openai.RequestError {
	return openai.New(http.StatusBadGateway, "upstream_error", "upstream_empty_response", "",
		"upstream returned an empty completion (no text) after retries; retry the request")
}

// writeEvent writes one SSE data frame.
func writeEvent(w http.ResponseWriter, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", b)
	return err
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
