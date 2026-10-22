package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"inceptionlabs2api/internal/openai"
	"inceptionlabs2api/internal/tools"
	"inceptionlabs2api/internal/upstream"
)

// toolPlan is the request-scoped state derived from tools/tool_choice.
type toolPlan struct {
	manifest *tools.Manifest
	hist     []openai.History
	// native reports that the gateway's own web_search tool was added to the
	// manifest, so its calls must be served internally rather than emitted.
	native bool
}

// planTools builds the manifest and history for a tool-bearing request. It
// returns a nil plan when the request must take the legacy path. native is the
// optional gateway-served search tool; it is added only when the client did
// not declare that name itself (a client tool always wins).
func planTools(req *openai.ChatRequest, native *tools.Tool) (*toolPlan, *openai.RequestError) {
	if len(req.Tools) == 0 || req.ToolChoice.Kind == openai.ToolChoiceNone {
		return nil, nil
	}
	list := make([]tools.Tool, 0, len(req.Tools)+1)
	for _, t := range req.Tools {
		if t.Function.Name == tools.NativeSearchName {
			native = nil
		}
		list = append(list, tools.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters})
	}
	if native != nil {
		list = append(list, *native)
	}
	choice := tools.Choice{}
	switch req.ToolChoice.Kind {
	case openai.ToolChoiceRequired:
		choice.Required = true
	case openai.ToolChoiceNamed:
		choice.Named = req.ToolChoice.Name
	default:
		choice.Auto = true
	}
	parallelOff := req.ParallelToolCalls != nil && !*req.ParallelToolCalls
	manifest, err := tools.BuildManifest(list, choice, parallelOff)
	if err != nil {
		return nil, openai.Invalid(err.Error())
	}
	hist, _, re := openai.NormalizeHistory(req.Messages)
	if re != nil {
		return nil, re
	}
	return &toolPlan{manifest: manifest, hist: hist, native: native != nil}, nil
}

// chatWithTools answers a request whose tools are active. A plain manifest
// streams; the native web_search loop resolves internally first, so a
// manifest with native search on takes the pre-buffered path.
func (s *Server) chatWithTools(w http.ResponseWriter, r *http.Request, req *openai.ChatRequest, plan *toolPlan) {
	ureq := upstream.NewToolChatRequestHistory(plan.hist, req.ReasoningEffort, plan.manifest)
	if req.Stream {
		s.streamTools(w, r, ureq, req.Model, plan.manifest, plan)
		return
	}
	s.completeTools(w, r, ureq, req.Model, plan.manifest, plan)
}

// round is one upstream response with its parsing applied.
type round struct {
	content   string
	reasoning string
	calls     []tools.Call
	cites     upstream.CitationCollector
}

// readRound opens one upstream stream and consumes it fully.
func (s *Server) readRound(ctx context.Context, ureq *upstream.ChatRequest, manifest *tools.Manifest) (*round, error) {
	stream, err := s.up.Chat(ctx, ureq, s.cfg.IdleTimeout, func(line string) {
		s.logf("upstream: skipping malformed SSE line: %q", truncate(line, 200))
	})
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var out round
	parser := tools.NewParser(manifest.Declared)
	rtag := wireTag()
	readErr := stream.Reader.Read(ctx, func(ev upstream.Event) error {
		out.cites.Observe(ev, s.logf)
		defer wireDumpEvent(rtag, ev, 0, 0)
		switch ev.Type {
		case upstream.EventReasoningDelta:
			out.reasoning += ev.Reasoning()
		case upstream.EventTextStart, upstream.EventTextDelta:
			out.content += string(parser.Feed([]byte(ev.ContentText())))
		}
		return nil
	})
	if readErr != nil {
		return nil, readErr
	}
	out.content += string(parser.Flush())
	out.calls = parser.Calls()
	return &out, nil
}

// emptyToolAnswer reports the measured upstream failure mode: with a tool
// manifest present the model occasionally returns reasoning-only, zero text
// and zero calls (probe/outQ q1/q3/q4, probe/outP2 timing runs). The upstream
// web surface shows the same flakiness without any preamble, so this is the
// model's occasional refusal-shaped pause, not a request defect. Reasoning
// without text or a call is the pause itself (the streaming path always
// treated it so via sawAlive), so only content and calls decide. The client
// asked for tool calling; an empty answer is not a valid tool response, so
// the gateway re-asks once internally with the same manifest — protocol
// completion, not content interception: whatever the retry returns is passed
// through verbatim, empty or not.
func (r *round) emptyToolAnswer() bool {
	return r.content == "" && len(r.calls) == 0
}

// retryKind classifies one zero-call round for the recovery retry. The empty
// answer keeps its own legacy branch (same-manifest re-ask); a broken-call
// round — truncated envelope or unparseable block — gets the ToolForge-style
// recovery retry; clean prose returns false: the gateway translates, never
// blocks, and a clean prose answer is a legal terminal.
func (s *Server) retryKind(res *round, calls []tools.Call) (tools.RecoveryKind, bool) {
	all := res.calls
	if calls != nil {
		all = calls
	}
	if len(all) > 0 {
		return tools.RecoveryClean, false
	}
	if res.emptyToolAnswer() {
		return tools.RecoveryClean, true // legacy empty branch
	}
	kind := tools.ClassifyNoCall(res.content, tools.ForeignOpenMarker)
	s.logf("tools: [recovery kind=%s] zero-call round (%d content bytes)", kind, len(res.content))
	return kind, kind != tools.RecoveryClean
}

// recoveryRequest clones ureq with the broken round appended as history —
// assistant(original content) + user(retry message) — so the model sees what
// it emitted and is asked to re-emit it properly.
func recoveryRequest(ureq *upstream.ChatRequest, res *round, message string) *upstream.ChatRequest {
	clone := *ureq
	msgs := make([]upstream.Message, 0, len(ureq.Messages)+2)
	msgs = append(msgs, ureq.Messages...)
	if res.content != "" {
		msgs = append(msgs, upstream.Message{
			Parts: []upstream.Part{{Type: "text", Text: res.content, State: "done"}},
			ID:    upstream.RandomID("m"),
			Role:  "assistant",
		})
	}
	msgs = append(msgs, upstream.Message{
		Parts: []upstream.Part{{Type: "text", Text: message}},
		ID:    upstream.RandomID("m"),
		Role:  "user",
	})
	clone.Messages = msgs
	return &clone
}

// recoverRound runs the ONE bounded recovery retry for a retry-worthy
// zero-call round: the empty answer re-asks with the same manifest (legacy
// behavior), the broken-call kinds re-ask with the original output appended
// as history plus the retry message. The retry result passes through
// verbatim — call or prose, empty or not; a transport failure keeps the
// ORIGINAL round rather than failing a client who already has an answer.
func (s *Server) recoverRound(ctx context.Context, ureq *upstream.ChatRequest, manifest *tools.Manifest, res *round, kind tools.RecoveryKind) *round {
	retryReq := recoveryRequestFor(ureq, res, kind)
	s.logf("tools: %s, retrying once", retryLogReason(kind, res))
	retry, err := s.readRound(ctx, retryReq, manifest)
	if err != nil {
		s.upErrors.Add(1)
		s.logf("tools: recovery retry failed: %v (passing original round through)", err)
		return res
	}
	return retry
}

// readRoundWithRetries applies the empty-answer policy on top of readRound:
// an empty round (no text, no calls) is retried EmptyRetries times because the
// upstream occasionally answers with nothing at all; a broken call envelope
// keeps its single recovery retry. The caller decides what to do when the
// result is still empty (measured 2026-10-22: mercury-3, 80k+ token prompts,
// completion_tokens=0 — the client saw a silent empty turn).
func (s *Server) readRoundWithRetries(ctx context.Context, ureq *upstream.ChatRequest, manifest *tools.Manifest) (*round, error) {
	res, err := s.readRound(ctx, ureq, manifest)
	if err != nil {
		return nil, err
	}
	emptyRetries := 0
	for {
		kind, doRetry := s.retryKind(res, nil)
		if !doRetry {
			return res, nil
		}
		if res.emptyToolAnswer() {
			if emptyRetries >= s.cfg.EmptyRetries {
				return res, nil
			}
			emptyRetries++
		}
		res = s.recoverRound(ctx, ureq, manifest, res, kind)
	}
}

// retryLogReason names the retry for the log line.
func retryLogReason(kind tools.RecoveryKind, res *round) string {
	if res.emptyToolAnswer() {
		return "empty answer round (no text/call)"
	}
	return fmt.Sprintf("[recovery kind=%s] broken-call round", kind)
}

// recoveryRequestFor builds the ureq a recovery retry re-asks with, shared by
// every retry path (stream/non-stream × plain/native). The empty answer
// keeps the same manifest (legacy); the broken-call kinds carry the original
// output appended as history plus the retry message.
func recoveryRequestFor(ureq *upstream.ChatRequest, res *round, kind tools.RecoveryKind) *upstream.ChatRequest {
	switch {
	case res.emptyToolAnswer():
		return ureq
	case kind == tools.RecoveryTruncated:
		return recoveryRequest(ureq, res, tools.RetryMessageTruncated(res.content))
	case kind == tools.RecoveryParseFailed:
		return recoveryRequest(ureq, res, tools.RetryMessageParseFailed(res.content))
	}
	return ureq
}

// emitToolCalls writes one delta frame per parsed call and returns the number
// emitted so far.
func emitToolCalls(w http.ResponseWriter, flusher http.Flusher, calls []tools.Call, emitted int, write func(openai.Delta, *string) bool) int {
	for ; emitted < len(calls); emitted++ {
		c := calls[emitted]
		delta := openai.Delta{ToolCalls: []openai.ToolCallDelta{{
			Index: emitted,
			ID:    tools.NewID(),
			Type:  "function",
			Function: openai.FunctionCallDelta{
				Name:      c.Name,
				Arguments: c.Arguments,
			},
		}}}
		if !write(delta, nil) {
			return emitted
		}
	}
	return emitted
}

// setStreamHeaders commits the SSE response headers shared by every streaming
// path (plain chat, tools, native). Headers are set only after the upstream
// stream is open and the round(s) are classified, so early failures still
// carry a proper HTTP status.
func setStreamHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
}

// streamTools is the streaming path for a plain manifest. With the
// native search tool active it runs the internal loop first: every round is
// consumed privately (no bytes) until the model stops calling web_search, and
// only that final round is replayed as frames.
//
// The plain path consumes each round fully before any frame leaves (the
// upstream sends whole answers in single events on this diffusion model, so
// nothing is lost). That is what makes the recovery retry sound: a zero-call
// round can be classified — clean prose, the measured empty answer, a
// truncated call envelope, an unparseable block — BEFORE the client has seen
// a byte, so a broken round never streams half its narration and the retry
// result (call or prose) is passed through verbatim.
func (s *Server) streamTools(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string, manifest *tools.Manifest, plan *toolPlan) {
	if plan.native {
		s.streamNative(w, r, ureq, model, plan)
		return
	}
	rtag := wireTag()
	res, err := s.readRoundWithRetries(r.Context(), ureq, manifest)
	if err != nil {
		s.fail(w, s.upstreamError(err))
		return
	}
	if res.emptyToolAnswer() {
		s.fail(w, emptyUpstreamError())
		return
	}
	wireDumpRound(rtag, res.content, res.calls)

	setStreamHeaders(w)
	flusher, _ := w.(http.Flusher)
	s.streams.Add(1)

	write := s.chunkWriter(w, flusher, upstream.RandomID("chatcmpl"), s.now().Unix(), model, rtag)
	if !write(openai.Delta{Role: "assistant"}, nil) {
		return
	}
	s.replayRound(w, flusher, write, res, res.calls)
}

// streamNative consumes rounds privately until the model stops calling the
// gateway's web_search tool, then replays the final round as frames. Nothing
// from an internal round is ever written, so the first client-visible frame
// is always part of the answer the client asked for.
func (s *Server) streamNative(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string, plan *toolPlan) {
	rtag := wireTag()
	res, calls, err := s.resolveNative(r.Context(), ureq, plan)
	if err != nil {
		if nativeAbort(r, err) {
			return
		}
		s.fail(w, s.upstreamError(err))
		return
	}
	wireDumpRound(rtag, res.content, calls)
	// The final round gets the same classification and one bounded retry as
	// the plain path. Nothing has been written yet, so the retry is free.
	if kind, doRetry := s.retryKind(res, calls); doRetry {
		if retryRes, retryCalls, err2 := s.resolveNative(r.Context(), recoveryRequestFor(ureq, res, kind), plan); err2 == nil {
			res, calls = retryRes, retryCalls
			wireDumpRound(rtag, res.content, calls)
		} else if nativeAbort(r, err2) {
			return
		} else {
			s.upErrors.Add(1)
			s.logf("tools: native recovery retry failed: %v (passing original round through)", err2)
		}
	}

	if res.content == "" && len(calls) == 0 {
		s.fail(w, emptyUpstreamError())
		return
	}

	setStreamHeaders(w)
	flusher, _ := w.(http.Flusher)
	s.streams.Add(1)

	write := s.chunkWriter(w, flusher, upstream.RandomID("chatcmpl"), s.now().Unix(), model, rtag)
	if !write(openai.Delta{Role: "assistant"}, nil) {
		return
	}
	s.replayRound(w, flusher, write, res, calls)
}

// resolveNative runs the internal web_search loop: read a round, serve every
// native call, append its result, and re-ask. It returns the first round with
// no native call plus any client calls accumulated from internal rounds (they
// are emitted with the final answer so no client tool call is lost).
func (s *Server) resolveNative(ctx context.Context, ureq *upstream.ChatRequest, plan *toolPlan) (*round, []tools.Call, error) {
	manifest := plan.manifest
	hist := plan.hist
	effort := ureq.ReasoningEffort
	budget := 0
	var pending []tools.Call
	// One round per allowed search, plus the round that reports the budget
	// exhaustion, plus a final round the model answers in. A model that keeps
	// calling past that is resolved with the calls stripped rather than looped.
	limit := nativeSearchBudget + 2
	var last *round
	for round := 0; round < limit; round++ {
		res, err := s.readRound(ctx, upstream.NewToolChatRequestHistory(hist, effort, manifest), manifest)
		if err != nil {
			return nil, nil, err
		}
		last = res
		native, client := nativeRound(res.calls)
		if len(native) == 0 {
			return res, append(pending, client...), nil
		}
		step, used, err := s.serveNativeCalls(ctx, native, res.content, budget)
		if err != nil {
			return nil, nil, err
		}
		budget = used
		hist = append(hist, step...)
		pending = append(pending, client...)
	}
	// The model ignored the budget and the round cap: answer with its last
	// round, with the unresolved native calls stripped. A refusal-by-loop is
	// not a client-visible error; the stream still closes normally.
	s.logf("tools: native search round cap reached; answering with the last round")
	return last, pending, nil
}

// replayRound writes an already-consumed round as the client-visible answer.
// The tool_calls finish reason is decided by the calls that remain after the
// native ones were served.
func (s *Server) replayRound(w http.ResponseWriter, flusher http.Flusher, write func(openai.Delta, *string) bool, res *round, calls []tools.Call) {
	if res.reasoning != "" && !write(openai.Delta{ReasoningContent: res.reasoning}, nil) {
		return
	}
	if res.content != "" && !write(openai.Delta{Content: res.content}, nil) {
		return
	}
	emitToolCalls(w, flusher, calls, 0, write)
	finish := "stop"
	if len(calls) > 0 {
		finish = "tool_calls"
	}
	if !write(openai.Delta{Annotations: res.cites.Citations()}, &finish) {
		return
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

// chunkWriter builds the frame writer for one streamed response.
func (s *Server) chunkWriter(w http.ResponseWriter, flusher http.Flusher, id string, created int64, model string, tag string) func(openai.Delta, *string) bool {
	return func(delta openai.Delta, finish *string) bool {
		chunk := openai.Chunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChunkChoice{{Index: 0, Delta: delta, FinishReason: finish}},
		}
		if err := writeEvent(w, chunk); err != nil {
			return false
		}
		wireDumpChunk(tag, chunk)
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
}

// completeTools is the non-streaming path for a plain manifest.
func (s *Server) completeTools(w http.ResponseWriter, r *http.Request, ureq *upstream.ChatRequest, model string, manifest *tools.Manifest, plan *toolPlan) {
	var res *round
	var calls []tools.Call
	var err error
	if plan.native {
		res, calls, err = s.resolveNative(r.Context(), ureq, plan)
	} else {
		res, err = s.readRoundWithRetries(r.Context(), ureq, manifest)
	}
	if err != nil {
		if errors.Is(err, context.Canceled) && r.Context().Err() != nil {
			return
		}
		s.fail(w, s.upstreamError(err))
		return
	}
	if plan.native {
		// The native loop's final round gets the same classification and one
		// bounded retry as the plain path. The retry re-enters the native
		// loop (not a bare readRound) so a retry round that calls web_search
		// is served internally rather than leaked to the client.
		if kind, doRetry := s.retryKind(res, calls); doRetry {
			if retryRes, retryCalls, err2 := s.resolveNative(r.Context(), recoveryRequestFor(ureq, res, kind), plan); err2 == nil {
				res, calls = retryRes, retryCalls
			} else {
				s.upErrors.Add(1)
				s.logf("tools: native recovery retry failed: %v (passing original round through)", err2)
			}
		}
		if res.content == "" && len(calls) == 0 {
			s.fail(w, emptyUpstreamError())
			return
		}
		s.writeToolCompletion(w, model, res.content, res.reasoning, calls, res.cites.Citations())
		return
	}
	if res.emptyToolAnswer() {
		s.fail(w, emptyUpstreamError())
		return
	}
	s.writeToolCompletion(w, model, res.content, res.reasoning, res.calls, res.cites.Citations())
}

// writeToolCompletion writes the aggregated chat.completion for tool output.
func (s *Server) writeToolCompletion(w http.ResponseWriter, model, content, reasoning string, calls []tools.Call, cites []openai.Citation) {
	msg := openai.CompletionMessage{
		Role:             "assistant",
		Content:          content,
		ReasoningContent: reasoning,
		Annotations:      cites,
	}
	finish := "stop"
	if len(calls) > 0 {
		msg.ToolCalls = make([]openai.ToolCall, 0, len(calls))
		for _, c := range calls {
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
				ID:       tools.NewID(),
				Type:     "function",
				Function: openai.FunctionCall{Name: c.Name, Arguments: c.Arguments},
			})
		}
		finish = "tool_calls"
	}
	writeJSON(w, http.StatusOK, openai.Completion{
		ID:      upstream.RandomID("chatcmpl"),
		Object:  "chat.completion",
		Created: s.now().Unix(),
		Model:   model,
		Choices: []openai.CompletionChoice{{
			Index:        0,
			Message:      msg,
			FinishReason: finish,
		}},
	})
}
