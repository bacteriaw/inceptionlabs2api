package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"inceptionlabs2api/internal/openai"
	"inceptionlabs2api/internal/upstream"
)

// This file restores the shared test helpers that the upstream publisher's
// "strip tests" pass (commit 545f5b9) removed while leaving two test files
// behind that still reference them. Minimal implementations matching the
// call sites in search_answer_test.go and history_fold_test.go.

// sseText renders one SSE round body whose single event carries text.
func sseText(text string) string {
	var b strings.Builder
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(upstream.Event{Type: upstream.EventTextStart, Snapshot: text})
	return "data: " + strings.TrimRight(b.String(), "\n") + "\n\ndata: [DONE]\n\n"
}

// mockChatter scripts one SSE body per upstream call (one "round" each).
type mockChatter struct {
	mu      sync.Mutex
	rounds  []string
	calls   int
	reqs    []*upstream.ChatRequest
}

func (m *mockChatter) Chat(ctx context.Context, req *upstream.ChatRequest, idle time.Duration, onMalformed func(string)) (*upstream.Stream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.calls >= len(m.rounds) {
		return nil, fmt.Errorf("mockChatter: no more scripted rounds (called %d times)", m.calls)
	}
	body := m.rounds[m.calls]
	m.reqs = append(m.reqs, req)
	m.calls++
	return &upstream.Stream{
		Body:   io.NopCloser(strings.NewReader(body)),
		Reader: upstream.NewReader(strings.NewReader(body), idle, nil),
	}, nil
}

func (m *mockChatter) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

// request returns the i-th upstream chat request the mock received.
func (m *mockChatter) request(i int) *upstream.ChatRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	if i < 0 || i >= len(m.reqs) {
		return nil
	}
	return m.reqs[i]
}


// partTexts extracts the text parts of an upstream message.
func partTexts(m upstream.Message) []string {
	var out []string
	for _, p := range m.Parts {
		if p.Text != "" {
			out = append(out, p.Text)
		}
	}
	return out
}

// postChat posts a chat-completions body and returns the recorder.
func postChat(t *testing.T, s *Server, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)
	return w
}


// weatherRequest builds the canonical get_weather chat request (stream flag
// selects SSE vs JSON), matching every weatherRequest(stream) call site.
func weatherRequest(stream bool) []byte {
	body := map[string]any{
		"model":    "mercury",
		"stream":   stream,
		"messages": []map[string]any{{"role": "user", "content": "weather in Nanning?"}},
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "get_weather",
				"description": "Get current weather",
				"parameters":  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
			},
		}},
	}
	b, _ := json.Marshal(body)
	return b
}

// streamFrames parses one SSE response body into its chat.completion.chunk
// frames (skipping the [DONE] sentinel).
func streamFrames(t *testing.T, body string) []openai.Chunk {
	t.Helper()
	var out []openai.Chunk
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var c openai.Chunk
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("bad SSE frame %q: %v", payload, err)
		}
		out = append(out, c)
	}
	return out
}


// newTestServer wires a bare Server (no native search), as used by the
// no-annotations and history-fold tests.
func newTestServer(t *testing.T, mock Chatter) *Server {
	t.Helper()
	return New(Config{Model: "mercury", Logf: func(string, ...any) {}}, mock, nil)
}
