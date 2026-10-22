package openai

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// TextMessage is a chat message reduced to plain text parts.
type TextMessage struct {
	Role  string
	Texts []string
}

// NormalizeMessages flattens every message to text parts. It rejects only a
// message that yields no renderable text at all: the upstream parts format
// cannot represent an empty message, and silently dropping it would change
// the conversation. Tool history is coerced to ordinary turns, unknown roles
// become "user", and media parts become short text placeholders — the
// gateway translates, it does not second-guess (§1.1).
//
// It returns the normalized messages and the total number of characters
// (informational: there is no client-side length guard).
func NormalizeMessages(msgs []Message) ([]TextMessage, int, *RequestError) {
	out := make([]TextMessage, 0, len(msgs))
	total := 0

	for i, m := range msgs {
		texts := contentTexts(m.Content)
		if len(texts) == 0 {
			return nil, 0, &RequestError{
				Status:  400,
				Type:    "invalid_request_error",
				Code:    CodeInvalidRequest,
				Param:   fmt.Sprintf("messages[%d].content", i),
				Message: fmt.Sprintf("messages[%d] (%s) has no text content", i, m.Role),
			}
		}
		for _, t := range texts {
			total += utf8.RuneCountInString(t)
		}
		out = append(out, TextMessage{Role: normalizeRole(m.Role), Texts: texts})
	}
	return out, total, nil
}

// normalizeRole maps OpenAI roles onto the three the upstream accepts. Tool
// and function results are text the caller supplied, so they ride as user
// turns; an unknown role is treated the same way rather than rejected.
func normalizeRole(role string) string {
	switch role {
	case "system", "developer":
		return "system"
	case "assistant":
		return "assistant"
	default:
		return "user"
	}
}

// ContentTexts exposes contentTexts to callers that render one message at a
// time (history compaction folds old turns into a single transcript bubble).
func ContentTexts(content any) []string { return contentTexts(content) }

// contentTexts extracts the renderable text of a message content value.
// A non-text part is rendered as a short placeholder naming its type, so an
// attachment stays visible to the model. Parts that carry no usable text
// simply contribute nothing.
func contentTexts(content any) []string {
	switch c := content.(type) {
	case nil:
		return nil
	case string:
		if c == "" {
			return nil
		}
		return []string{c}
	case []any:
		texts := make([]string, 0, len(c))
		for _, item := range c {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			typ, _ := part["type"].(string)
			if typ == "text" {
				if text, ok := part["text"].(string); ok && text != "" {
					texts = append(texts, text)
				}
				continue
			}
			if placeholder := mediaPlaceholder(typ); placeholder != "" {
				texts = append(texts, placeholder)
			}
		}
		return texts
	default:
		return nil
	}
}

// mediaPlaceholder is the text stand-in for a non-text content part.
func mediaPlaceholder(typ string) string {
	switch typ {
	case "image_url", "image":
		return "[image]"
	case "input_audio", "audio":
		return "[audio]"
	case "file", "document":
		return "[file]"
	case "video_url":
		return "[video_url]"
	case "video":
		return "[video]"
	case "":
		return ""
	}
	// Unknown future part types still carry intent; show that one was sent
	// without echoing unbounded attacker-controlled type strings.
	if len(typ) > 32 {
		typ = typ[:32]
	}
	return "[" + strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, typ) + "]"
}
