package server

import (
	"inceptionlabs2api/internal/openai"
)

// trimMessages caps the message count because the upstream web chat rejects
// any conversation with more than 100 messages:
//
//	{"error":"This conversation is too long. Please start a new chat to continue."}
//
// Measured 2026-10-22: 100 messages pass, 101 fail, independent of the payload
// size (a 900 KB single message is accepted), so this is a message-count cap,
// not a token cap.
//
// The trim keeps every leading system/developer message plus the newest turns,
// and drops old turns in whole groups so an assistant tool_calls message is
// never separated from the tool results that answer it. It returns the number
// of dropped messages so the caller can log it.
func trimMessages(msgs []openai.Message, max int) ([]openai.Message, int) {
	if max <= 0 || len(msgs) <= max {
		return msgs, 0
	}

	head := 0
	for head < len(msgs) && (msgs[head].Role == "system" || msgs[head].Role == "developer") {
		head++
	}
	if head > max { // pathological: system preamble alone is over budget
		head = max
	}

	// Group the tail: a message that carries tool_calls owns the tool results
	// that follow it, so the whole span is kept or dropped together.
	type span struct{ start, end int } // [start,end)
	var groups []span
	for i := head; i < len(msgs); {
		j := i + 1
		if hasToolCalls(msgs[i]) {
			for j < len(msgs) && msgs[j].Role == "tool" {
				j++
			}
		}
		groups = append(groups, span{i, j})
		i = j
	}

	// Fill the budget from the newest group backwards.
	budget := max - head
	used := 0
	keepIdx := len(groups)
	for i := len(groups) - 1; i >= 0; i-- {
		size := groups[i].end - groups[i].start
		if used+size > budget {
			break
		}
		used += size
		keepIdx = i
	}
	if keepIdx >= len(groups) {
		// Even the newest group alone does not fit: keep just that group.
		keepIdx = len(groups) - 1
	}

	keepFrom := groups[keepIdx].start
	out := make([]openai.Message, 0, head+len(msgs)-keepFrom)
	out = append(out, msgs[:head]...)
	out = append(out, msgs[keepFrom:]...)

	// Drop leading non-user turns left over from the cut so the upstream sees
	// a prompt first, then a normal alternating conversation.
	for len(out) > head && out[head].Role != "user" {
		out = append(out[:head], out[head+1:]...)
	}
	return out, len(msgs) - len(out)
}

func hasToolCalls(m openai.Message) bool {
	return len(m.ToolCalls) > 0 && string(m.ToolCalls) != "null"
}
