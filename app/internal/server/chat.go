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

// stream writes the OpenAI SSE response with a DELAYED header commit: the
// SSE headers and the role frame are not written until the first visible
// text delta arrives, and every delta seen before then (reasoning deltas
// included) is buffered in memory and replayed in upstream order once the
// commit fires. An empty completion — the stream ends with zero visible
// text — therefore surfaces as a real HTTP 502 via the error envelope
// instead of a 200 with an empty SSE body; a mid-stream failure before any
// text surfaces as its mapped status. Once text has arrived the response is
// a committed SSE stream, and later failures ride the in-stream error frame
// as before. The frame order of a non-empty stream is unchanged: role,
// buffered reasoning, text, finish.
func (s *Server) stream(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string) {
	rtag := wireTag()
	wireLog(rtag, "PLAIN STREAM (no tools path)")
	stream, err := s.up.Chat(r.Context(), ureq, s.cfg.IdleTimeout, func(line string) {
		s.logf("upstream: skipping malformed SSE line: %q", truncate(line, 200))
	})
	if err != nil {
		s.fail(w, s.upstreamError(err))
		return
	}
	defer stream.Close()

	// delayed is the writer state until the first visible text arrives: the
	// headers are not committed, and reasoning deltas buffer in order. The
	// upstream sends whole answers in single events on this diffusion model,
	// so the buffer is one short reasoning string in practice.
	var (
		committed  bool
		buffered   []openai.Delta
		id         = upstream.RandomID("chatcmpl")
		created    = s.now().Unix()
		flusher    http.Flusher
		hasFlusher bool
	)

	write := func(delta openai.Delta, finish *string) bool {
		if !committed {
			return true // pre-commit writes buffer via the read loop, not here
		}
		chunk := openai.Chunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChunkChoice{{Index: 0, Delta: delta, FinishReason: finish}},
		}
		if err := writeEvent(w, chunk); err != nil {
			return false
		}
		if hasFlusher {
			flusher.Flush()
		}
		return true
	}

	// commit writes the SSE headers, the role frame, and every buffered
	// delta in original order. It fires exactly once, on the first visible
	// text delta.
	commit := func() bool {
		if committed {
			return true
		}
		committed = true
		setStreamHeaders(w)
		s.streams.Add(1)
		flusher, hasFlusher = w.(http.Flusher)
		if !write(openai.Delta{Role: "assistant"}, nil) {
			return false
		}
		for _, d := range buffered {
			if !write(d, nil) {
				return false
			}
		}
		buffered = nil
		return true
	}

	// Sources are collected as they arrive and ride the finish frame
	// (SPEC §2.4): the upstream interleaves text-end → sources → text-start,
	// so no earlier chunk is known to be the last one.
	var cites upstream.CitationCollector
	sawText := false
	readErr := stream.Reader.Read(r.Context(), func(ev upstream.Event) error {
		cites.Observe(ev, s.logf)
		if wireDebugOn() {
			wireDumpEvent(rtag, ev, 0, 0)
		}
		switch ev.Type {
		case upstream.EventReasoningDelta:
			if text := ev.Reasoning(); text != "" {
				if committed {
					if !write(openai.Delta{ReasoningContent: text}, nil) {
						return errClientGone
					}
				} else {
					// Buffer only: the header commit waits for visible text.
					buffered = append(buffered, openai.Delta{ReasoningContent: text})
				}
			}
		case upstream.EventTextStart, upstream.EventTextDelta:
			if text := ev.ContentText(); text != "" {
				if !sawText {
					sawText = true
					if !commit() {
						return errClientGone
					}
				}
				if !write(openai.Delta{Content: text}, nil) {
					return errClientGone
				}
			}
		}
		return nil
	})

	if readErr != nil {
		if errors.Is(readErr, errClientGone) {
			return
		}
		if errors.Is(readErr, context.Canceled) && r.Context().Err() != nil {
			// Client went away: nothing to write to, and the pool token
			// stays valid.
			return
		}
		s.upErrors.Add(1)
		if !committed {
			// The failure happened before any visible text: the client has
			// seen nothing, so it gets a real HTTP status for it.
			s.logf("upstream: stream failed before any text: %v", readErr)
			s.fail(w, s.upstreamError(readErr))
			return
		}
		s.logf("upstream: stream failed after headers: %v", readErr)
		se := streamError(readErr)
		_ = writeEvent(w, se.Envelope())
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if hasFlusher {
			flusher.Flush()
		}
		return
	}

	if !sawText {
		// The empty completion: the stream finished cleanly but never
		// produced a visible text delta. Nothing has been written, so the
		// client gets a proper 502 rather than a 200 empty SSE stream — the
		// same shape the non-streaming empty path returns.
		s.logf("upstream: plain stream ended with no visible text")
		s.fail(w, s.upstreamError(errUpstreamEmpty))
		return
	}

	stop := "stop"
	if !write(openai.Delta{Annotations: cites.Citations()}, &stop) {
		return
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if hasFlusher {
		flusher.Flush()
	}
}

// errClientGone aborts the read loop when writing to the client fails.
var errClientGone = errors.New("server: client write failed")

// complete aggregates the upstream stream into a single chat.completion.
func (s *Server) complete(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string) {
	stream, err := s.up.Chat(r.Context(), ureq, s.cfg.IdleTimeout, func(line string) {
		s.logf("upstream: skipping malformed SSE line: %q", truncate(line, 200))
	})
	if err != nil {
		s.fail(w, s.upstreamError(err))
		return
	}
	defer stream.Close()

	var content, reasoning strings.Builder
	var cites upstream.CitationCollector
	readErr := stream.Reader.Read(r.Context(), func(ev upstream.Event) error {
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
		if errors.Is(readErr, context.Canceled) && r.Context().Err() != nil {
			return
		}
		s.fail(w, s.upstreamError(readErr))
		return
	}

	writeJSON(w, http.StatusOK, openai.Completion{
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
	})
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
