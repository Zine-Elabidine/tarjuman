# The Tarjuman format (v1, 2026-09-29)

Status: **locked and implemented** (decisions in §8). This document checks the v0 format
against every protocol we plan to support and against what Diwan and a future gateway need,
then describes v1.

Checked against: OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Gemini, the
"compatible" dialects (DeepSeek, OpenRouter, vLLM), and the two references that solved the
same problem: DeepSeek Harness `packages/llm/llm/src/{types,message}.ts` and pi-ai
`packages/ai/src/types.ts` + `api/transform-messages.ts`.

---

## 1. What the format is for (and what it is not)

The format is **exactly what a model sees, plus what is needed to replay it losslessly**.

- It **is**: messages, blocks, tool definitions, a request, stream events, usage, errors.
- It **is not** the session log. Who produced a message (the user, a hook, a memory recall,
  another agent), timestamps, event ids, branches: all of that lives in **Diwan's event**
  that wraps the message. DeepSeek Harness puts `source` on the message itself; we keep
  Tarjuman free of runtime concepts so it stays usable alone and behind a gateway.
- Everything is plain JSON data with a version, because it crosses three boundaries: the
  Diwan log on disk, the process boundary (gateway), and time (old sessions must still load).

---

## 2. Where the protocols disagree (the things the format must survive)

| Concern | OpenAI Chat | OpenAI Responses | Anthropic | Gemini |
|---|---|---|---|---|
| System prompt | message, role `system`/`developer`, anywhere | `instructions` + `developer` items | top-level `system` only | top-level `systemInstruction` only |
| Reasoning | not standard; `reasoning_content` (DeepSeek), `reasoning` (OpenRouter) | `reasoning` item: summary + `encrypted_content`, has an id | `thinking` + **signature**; `redacted_thinking` (opaque `data`) | parts with `thought: true` + `thoughtSignature` |
| Must reasoning be sent back? | DeepSeek thinking+tools: **yes, within the turn** | yes (encrypted, when `store=false`) | **yes, within a tool loop**, signature intact | signatures yes |
| Tool call | JSON **string** args, id `call_*` | `function_call` item: `id` **and** `call_id` | `tool_use`, args **object**, id `^[a-zA-Z0-9_-]{1,64}$` | `functionCall`, args object, id optional, **signature on the part** |
| Tool result | role `tool`, one message per call, text only (images not allowed) | `function_call_output` item | block in a **user** message, text + images, `is_error` | `functionResponse` needs the **function name** |
| Images | `image_url` (url or data URI) | `input_image` | `image` block (base64 / url) | `inlineData` / `fileData` |
| Stop reasons | `stop`, `length`, `tool_calls`, `content_filter` | status + incomplete reason | `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`, `pause_turn`, `refusal`, context exceeded | `STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`, `MALFORMED_FUNCTION_CALL` |
| Usage | `prompt_tokens` **includes** cache hits | same | `input_tokens` **excludes** cache; read/write separate (5 min / 1 h) | `promptTokenCount` includes cached; thoughts separate |
| Refusal | `refusal` field | refusal content part | `stop_reason: refusal` | `SAFETY` |
| Role order | free | free | consecutive same roles get merged; first must be user | `user` / `model`, merged |
| Server-side tools | no | web search, file search, code interpreter items | `server_tool_use` + results | grounding, code execution |

---

## 3. Review of v0: what holds, what is missing, what is wrong

**Holds (keep):**
- Messages = role + list of typed blocks; `Text`, `Reasoning`, `ToolCall`, `ToolResult`.
- Tool arguments kept as the **raw string** the model wrote (may be invalid JSON; the loop
  reports it back instead of crashing).
- Disjoint usage (`input` = uncached only), `cost` when the provider reports it.
- Assistant messages carry `provider`, `model`, `usage`, `stop`.
- A `tool` message may hold several results (Anthropic groups them; OpenAI Chat splits them
  in the translator).

**Missing:**
1. **The replay envelope.** Without it, Anthropic signatures, redacted thinking, OpenAI
   encrypted reasoning, Responses item ids and Gemini thought signatures are all lost.
   Anthropic with thinking + tools **fails outright** if the signature is not sent back.
2. **`protocol` on assistant messages.** The same-model test is provider + protocol + model;
   OpenRouter can serve one model through two protocols with different replay data.
3. **Tool name on `ToolResult`.** Gemini's `functionResponse` requires it.
4. **Rich tool results.** Results are `str` only; screenshots and images need text + image blocks.
5. **Images** (user messages and tool results).
6. **Redacted/encrypted reasoning** has no representation.
7. **A request type.** Today it is `stream(model, messages, tools, max_tokens, **extra)`.
   Reasoning level, tool choice, stop sequences, temperature, cache hints and structured
   output have no neutral names, and a gateway needs the request as data.
8. **Serialization version** and forward compatibility (unknown block types from a newer
   version must survive a load/save round trip, not crash).
9. **More stop reasons:** `refusal`, `error`, `pause` (Anthropic `pause_turn`: the server
   paused a long server-tool turn; the caller must send it back to continue).

**Wrong / to fix:**
- `_Parser` overwrites `model` with whatever the chunk reports. OpenRouter reports the
  resolved model (e.g. a dated snapshot), which then breaks the same-model check. Keep the
  **requested** model in `model` and put the reported one in `response_model`.
- The OpenAI Chat translator never sends reasoning back. Correct for OpenAI, **wrong for
  DeepSeek thinking mode with tools** (needs `reasoning_content` on assistant messages of the
  current turn). That is a quirk flag, but the format must keep reasoning so the flag can work.

---

## 4. The proposed format

### Blocks

```python
Text(text, citations=None)             # citations: opaque, provider-specific, optional
Reasoning(text, redacted=False)        # redacted=True: text is empty, payload only in replay
                                       # summary=True when the provider only gave a summary (OpenAI)
Image(mime, data=None, url=None, ref=None)
                                       # exactly one of: base64 data, a URL, or a caller ref
                                       # ("sha256:..."), resolved by a loader passed to stream()
ToolCall(id, name, arguments)          # arguments: raw string, unchanged
ToolResult(call_id, name, content, is_error=False)
                                       # content: list[Text | Image]; str accepted and wrapped
Unknown(type, data)                    # a block from a newer version or an unmodelled provider
                                       # block: kept on round trip, never sent to a model
```

**Where blocks may appear** (enforced by a validator, not by the type system):

| Role | Allowed blocks |
|---|---|
| `system` | `Text` |
| `user` | `Text`, `Image` |
| `assistant` | `Text`, `Reasoning`, `ToolCall` |
| `tool` | `ToolResult` (one or more) |

Deliberately **not** blocks (for now): files/PDFs (the caller turns them into text or a path
the model can `read`, as DeepSeek Harness does), audio, server-side tool calls (see §7).

### Messages

```python
Message(
    role,                  # "system" | "user" | "assistant" | "tool"
    content,               # list[Block]
    # assistant only:
    provider=None,         # "anthropic", "openrouter", ...
    protocol=None,         # "anthropic-messages", "openai-chat", ...        (NEW)
    model=None,            # the model id we REQUESTED
    response_model=None,   # the model the provider says it used, if different  (NEW)
    usage=None,
    stop=None,
    replay=None,           # Replay, opaque                                   (NEW)
    error=None,            # message text when stop == "error"                (NEW)
    partial=None,          # interrupted: blocks cut off mid-stream; kept, never sent (NEW)
)

Replay(response=..., blocks=[...])
    # response: provider-private, response level (response id, native stop reason, ...)
    # blocks:   one entry per content block, same order (signature, encrypted payload,
    #           item id, thought signature...), None where a block has nothing
    # Only read when provider + protocol + model all match. If a block is dropped, its entry
    # is dropped with it; a length mismatch discards the whole envelope (never misalign).
```

- **System messages anywhere.** The first is the system prompt; later ones are instructions
  added from that point on, rendered as `<system-reminder>` text in the neighbouring
  tool-result or user message (§8, decision 2). The prompt prefix stays append-only
  (Diwan decision #9).
- **Any role order is legal in the format.** Merging consecutive same-role messages and
  "first message must be user" are translator jobs.

### Stop reasons

`end` · `tool_use` · `max_tokens` · `refusal` · `pause` · `interrupted` · `error`
(`content_filter` is folded into `refusal`; the native reason is kept in `replay.response`.)

### Usage

```python
Usage(input, cache_read, cache_write, output, reasoning, cost=None,
      cache_write_long=0)   # subset of cache_write written with the long (1 h) TTL; priced higher
```
Rules: `input` excludes cache; `output` includes `reasoning`; `cost` in USD from the provider,
else computed from the catalog, else `None`: never a guess.

### Tools

```python
Tool(name, description, parameters, strict=False)
# name: ^[a-zA-Z0-9_-]{1,64}$ (the strictest provider's rule; validated when defined)
# parameters: JSON Schema; translators downgrade it for Gemini's OpenAPI subset
```

### The request

```python
Request(
    model, messages, tools=None,
    tool_choice="auto",          # "auto" | "none" | "required" | {"name": ...}
    max_tokens=None,
    reasoning=None,              # None = provider default | "off" | "minimal" | "low" | "medium" | "high" | "max"
    temperature=None, top_p=None, stop=None,
    response_format=None,        # a JSON Schema for structured output
    cache="short",               # "none" | "short" | "long": a hint, mapped per provider
    session_id=None,             # for providers that route or cache by session
    extra=None,                  # raw fields merged into the wire body, last (escape hatch)
)
```
`stream(request)` is the one entry point (plus `complete(request)`). The current
`stream(model, messages, ...)` stays as a thin shortcut.

### Stream events

Unchanged vocabulary, two additions:
- `BlockEnd(index, block, replay)` carries the **assembled block and its replay entry** (a UI or
  gateway can use it without replaying deltas, and an interrupted caller keeps every finished
  block exactly, signature included).
- `Finish(message)` stays the last event; the message carries `replay`.

Errors: the library **raises** `TarjumanError` (Pythonic). Over a gateway the same error is
sent as a final `{"type": "error", code, message, status, retry_after}` event and raised
again on the client, so both paths look the same to Diwan.

### Serialization

```json
{"v": 1, "role": "assistant", "content": [...], "provider": "anthropic", ...}
```
- Every type has `to_dict` / `from_dict`; `from_dict(to_dict(x)) == x` is a test for every type.
- `v` on messages and requests. A loader reads every older `v`; an unknown block type becomes
  `Unknown` and is written back unchanged.
- Image `data` can be large: Diwan should store images as `ref` (content-addressed blobs next
  to the log) and pass a loader, so the JSONL log stays small.

---

## 5. The transform (runs before every request, for the target model)

pi-ai's rules plus the gaps above, in order:

1. **Validate** block placement per role; drop `Unknown` blocks.
2. **Same model?** provider + protocol + model equal → keep `replay` and reasoning as is.
3. **Different model:** drop `replay`; reasoning → `Text` (empty reasoning dropped); redacted
   reasoning dropped; citations dropped.
4. **`interrupted` assistant messages** send their finished `content` like any other message;
   `partial` (the block cut off mid-stream) is never sent (§8, decision 4).
   **`error`** assistant messages are skipped, and so are results of their tool calls.
5. **Tool-call ids** rewritten when the target's rule rejects them; results follow the map.
6. **Images** → placeholder text when the target has no vision (also inside tool results).
   Tool results with images → for OpenAI Chat, the text goes in the tool message and the
   image in a following user message (the usual workaround).
7. **Orphaned tool calls** get a synthetic error result (done in v0). System messages that
   land between a call and its results are held until after the results.
8. **Empty text blocks and empty messages** are dropped (Anthropic rejects them).

Pure function, no I/O, deterministic: the same history and target always give the same
request, which keeps prompt caches warm and makes replay testable.

---

## 6. What the translators own (not the format)

Folding system messages; merging same-role messages; where cache breakpoints go (Anthropic
`cache_control`); reasoning dialects (`reasoning_effort`, `thinking: {type, budget}`,
`enable_thinking`...); `max_tokens` field names; sending reasoning back (DeepSeek); schema
downgrade (Gemini); splitting grouped tool results (OpenAI Chat). All driven by the compat table.

---

## 7. Deliberately left out (and how to add each later without breaking the format)

| Left out | Why | Path later |
|---|---|---|
| `n > 1` choices | councils make separate calls; one request → one message | none needed |
| Files / PDFs as native blocks | only some providers accept them; text/path works everywhere | a `Document` block + placeholder rule |
| Audio in/out | not needed for a coding agent | new block types |
| Server-side tools (provider web search, code execution) | provider-specific, not replayable to others | `Unknown` blocks with a `Text` fallback, same-model replay only |
| Tool list changes mid-session (Anthropic `defer_loading`) | an optimization; decision #6 | system messages with `tools_added` / `tools_removed`, as pi-ai does |
| Stateful APIs (`previous_response_id`) | we always send the full history; the log is the truth | a replay hint only |

---

## 8. Decisions (2026-09-29)

1. ✅ **`Unknown` blocks are kept.** A block from a newer version (or an unmodelled provider
   block) loads as `Unknown(type, data)`, is saved back unchanged, and is never sent to a
   model. Needed because two machines and a gateway will run different versions.
2. ✅ **Mid-conversation system messages never touch the prompt prefix.** The leading
   `system` message goes to the provider's top-level slot; every later one is rendered as a
   `<system-reminder>…</system-reminder>` text block **appended to the pending tool-result
   message** if a tool loop is open, **else prepended to the next user message**, else held
   until one exists. This is the default for every protocol (appending is cache-safe
   everywhere); a compat flag may allow a native in-place `system`/`developer` message where
   that is proven cache-safe. Same approach Claude Code shows from the outside: its
   transcript stores reminders as separate `attachment` events with a `rendered` field, and
   merges them into the neighbouring user / tool-result turn at request time.
3. ✅ **Neutral reasoning levels:** `off | minimal | low | medium | high | max`; omitted =
   provider default; the catalog maps each level per model; an unsupported level falls to
   the nearest supported one.
4. ✅ **Interrupted turns: keep finished blocks, set the cut-off one apart, add a marker**
   (option D, what both Codex and Claude Code do). Blocks that received `BlockEnd` are
   complete (an Anthropic thinking block has its signature by then) and go in `content`
   with their replay entries; they are sent back as usual. The block cut off mid-stream goes
   in `Message.partial`: kept in the log, never sent. A finished tool call that never ran
   gets the usual synthetic result. The caller adds an "interrupted on purpose" `system`
   message, which is sent as a reminder. Evidence: Codex records history only on
   `OutputItemDone` and appends `<turn_aborted>`; Claude Code transcripts hold no unsigned
   thinking block (0 of 1,108) and append `[Request interrupted by user]`.
   Rejected: A (text only: loses signed reasoning), B (skip: the model loses the context),
   C (native partial blocks: per-provider rules, rejected requests).

**Rule for Diwan (from 2):** a context injection (budget warning, model switch, tools
changed, instructions changed, date) is **its own event** in the log, never an edit to a
user message, and the event stores the **exact rendered text**, not only its data. A resumed
or replayed session then rebuilds a byte-identical request (cache hits, exact replay) even
after the wording in the code has changed.

## 9. Migration from v0

Small: add fields with defaults (`protocol`, `response_model`, `replay`, `error`,
`ToolResult.name`), accept `str` for `ToolResult.content`, add `Image`/`Unknown`, add `v` to
`to_dict` and read v0 dicts (no `v`) as v0. Diwan's existing session logs keep loading.
Fix the `_Parser` model overwrite. Tests: round trip every type, transform rules one by one,
a recorded stream per protocol.
