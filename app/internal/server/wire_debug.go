package server

import (
	"encoding/json"
	"fmt"
	"inceptionlabs2api/internal/openai"
	"inceptionlabs2api/internal/tools"
	"log"
	"strings"
	"sync/atomic"

	"inceptionlabs2api/internal/upstream"
)

// Wire debug tap (IL2API_WIRE_DEBUG=on). Off by default: zero cost otherwise.
// Logs the client request body, every upstream event, and the tool parser's
// view of the content stream — the minimal evidence needed to tell a client
// loop bug from a gateway decode bug in a streaming multi-round session.

var wireDebug atomic.Bool
var wireSeq atomic.Int64

// Wire debug is disabled unconditionally: the tap was a live-incident tool,
// but per-event logging on the streaming hot path costs latency and log
// volume in production. IL2API_WIRE_DEBUG no longer re-enables it.
func init() { wireDebug.Store(false) }

// wireLog writes one line to stderr with a request tag so concurrent streams
// can be told apart.
func wireLog(tag, format string, args ...any) {
	if !wireDebug.Load() {
		return
	}
	log.SetFlags(log.Lmicroseconds)
	log.Printf("[wire %s] %s", tag, fmt.Sprintf(format, args...))
}

// wireDumpRequest logs the raw client request body (truncated per message).
func wireDumpRequest(tag string, body []byte) {
	if !wireDebug.Load() {
		return
	}
	var pretty map[string]any
	_ = json.Unmarshal(body, &pretty)
	if msgs, ok := pretty["messages"].([]any); ok {
		for i, m := range msgs {
			if mm, ok := m.(map[string]any); ok {
				for k, v := range mm {
					s := fmt.Sprintf("%v", v)
					if len(s) > 300 {
						s = s[:300] + "...(" + fmt.Sprint(len(s)) + " chars)"
					}
					mm[k] = s
				}
				msgs[i] = mm
			}
		}
		pretty["messages"] = msgs
	}
	b, _ := json.MarshalIndent(pretty, "", " ")
	wireLog(tag, "CLIENT REQUEST:\n%s", b)
}

// wireDumpEvent logs one upstream event as it crosses the relay.
func wireDumpEvent(tag string, ev upstream.Event, parserReleasable, parserCalls int) {
	if !wireDebug.Load() {
		return
	}
	detail := ""
	switch ev.Type {
	case upstream.EventReasoningDelta:
		detail = truncate(ev.Reasoning(), 120)
	case upstream.EventTextStart, upstream.EventTextDelta:
		detail = strings.ReplaceAll(truncate(ev.ContentText(), 160), "\n", "\\n")
	}
	wireLog(tag, "UPSTREAM %-18s %q releasable=%d calls=%d", ev.Type, detail, parserReleasable, parserCalls)
}

// wireDumpChunk logs one client-visible SSE chunk as it is written.
func wireDumpChunk(tag string, chunk openai.Chunk) {
	if !wireDebug.Load() {
		return
	}
	d := chunk.Choices[0].Delta
	parts := make([]string, 0, 3)
	if d.Role != "" {
		parts = append(parts, "role="+d.Role)
	}
	if d.Content != "" {
		parts = append(parts, fmt.Sprintf("content=%q", truncate(d.Content, 160)))
	}
	if len(d.ToolCalls) > 0 {
		t := d.ToolCalls[0]
		parts = append(parts, fmt.Sprintf("tool_call[%d]=%s(%s)", t.Index, t.Function.Name, truncate(t.Function.Arguments, 200)))
	}
	if chunk.Choices[0].FinishReason != nil {
		parts = append(parts, "finish="+*chunk.Choices[0].FinishReason)
	}
	if len(d.Annotations) > 0 {
		parts = append(parts, fmt.Sprintf("annotations=%d", len(d.Annotations)))
	}
	wireLog(tag, "CLIENT <- %s", strings.Join(parts, " "))
}

// wireDumpRound logs the parsed outcome of one consumed round (calls + text
// after Flush) — the ground truth of what the parser decided.
func wireDumpRound(tag string, content string, calls []tools.Call) {
	if !wireDebug.Load() {
		return
	}
	if content != "" {
		wireLog(tag, "ROUND text=%q", truncate(content, 200))
	}
	for i, c := range calls {
		wireLog(tag, "ROUND call[%d]=%s(%s)", i, c.Name, truncate(c.Arguments, 250))
	}
}

// wireTag returns a short per-request tag for correlating log lines.
func wireTag() string { return fmt.Sprintf("r%d", wireSeq.Add(1)) }

// wireDebugOn reports whether the wire tap is enabled. Callers in hot loops
// use this to skip defer bookkeeping entirely (per-event defer records pile
// up until the enclosing function returns — measurable on long streams).
func wireDebugOn() bool { return wireDebug.Load() }
