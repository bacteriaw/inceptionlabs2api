package server

import (
	"encoding/json"
	"fmt"
	"strings"

	"inceptionlabs2api/internal/openai"
)

// compactMessages folds the oldest turns into a single user message so a long
// conversation still fits the upstream's 100-message cap without losing text.
//
// Measured 2026-10-22: the upstream rejects the 101st message regardless of
// payload size (a 900 KB message passes), so message COUNT is the scarce
// resource. Trimming throws old turns away; compaction keeps their text and
// spends one slot on it instead of one slot per turn.
//
// Layout: [merged system] + [flattened history] + [last keepRecent messages].
// The tail never starts inside a tool group (an assistant tool_calls message
// stays with the tool results that answer it). Only the prefix is flattened;
// recent turns keep their real roles so the model's turn structure is intact.
//
// Measured 2026-10-22: the upstream accepts consecutive same-role messages and
// a flattened transcript bubble; a needle placed in folded history was still
// recalled (chat + tool-history variants, 3/3).
//
// It returns the compacted slice and the number of messages folded away.
func compactMessages(msgs []openai.Message, keepRecent, max int) ([]openai.Message, int) {
	if max <= 0 || len(msgs) <= max {
		return msgs, 0
	}
	if keepRecent <= 0 {
		keepRecent = max - 8
	}
	if keepRecent > max-2 {
		keepRecent = max - 2
	}
	if keepRecent < 2 {
		return msgs, 0
	}

	head := 0
	var systemTexts []string
	for head < len(msgs) && (msgs[head].Role == "system" || msgs[head].Role == "developer") {
		systemTexts = append(systemTexts, openai.ContentTexts(msgs[head].Content)...)
		head++
	}

	start := len(msgs) - keepRecent
	if start < head {
		return msgs, 0
	}
	// Never split a call from its results: a tail that begins with tool output
	// must drag its assistant tool_calls message along.
	for start > head && msgs[start].Role == "tool" {
		start--
	}
	if start <= head+1 {
		// One foldable message is not worth losing role structure for.
		return msgs, 0
	}

	prefix := msgs[head:start]
	var b strings.Builder
	b.WriteString("以下是本次对话更早部分的记录（为适配上游消息条数上限合并呈现，内容按原顺序完整保留）：\n\n")
	for _, m := range prefix {
		b.WriteString(renderTranscriptLine(m))
	}

	out := make([]openai.Message, 0, 2+len(msgs)-start)
	if len(systemTexts) > 0 {
		out = append(out, openai.Message{Role: "system", Content: strings.Join(systemTexts, "\n\n")})
	}
	out = append(out, openai.Message{Role: "user", Content: b.String()})
	out = append(out, msgs[start:]...)
	return out, len(prefix)
}

// renderTranscriptLine renders one folded message as a single transcript line.
func renderTranscriptLine(m openai.Message) string {
	role := m.Role
	if role == "developer" {
		role = "system"
	}
	var parts []string
	for _, t := range openai.ContentTexts(m.Content) {
		parts = append(parts, t)
	}
	if calls := toolCallSummary(m.ToolCalls); calls != "" {
		parts = append(parts, calls)
	}
	if len(parts) == 0 {
		return ""
	}
	label := role
	switch role {
	case "system":
		label = "系统"
	case "user":
		label = "用户"
	case "assistant":
		label = "助手"
	case "tool":
		label = "工具结果"
	}
	return fmt.Sprintf("%s：%s\n\n", label, strings.Join(parts, "；"))
}

// toolCallSummary renders an assistant tool_calls payload as short text.
func toolCallSummary(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var calls []struct {
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &calls); err != nil {
		return ""
	}
	rendered := make([]string, 0, len(calls))
	for _, c := range calls {
		rendered = append(rendered, fmt.Sprintf("调用 %s(%s)", c.Function.Name, c.Function.Arguments))
	}
	return strings.Join(rendered, "，")
}
