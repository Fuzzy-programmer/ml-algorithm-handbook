# Chapter 6.9 — Tool Calling (Function Calling)

[← Section index](README.md) · Prev [6.8](08-multimodal.md) · Next [6.10 Agents](10-agents-agentic-ai.md) · Priority 🔴 · Code: [`agent_loop_mock.py`](../code/agent_loop_mock.py) (executed, scripted model)

> **Version-sensitivity:** API field names and model behavior change. The SDK snippets below follow the Anthropic Python SDK reference as of late 2026 (model IDs cached 2026-10-06) and were **not executed here** (no API key in the authoring environment). The control-flow code in `agent_loop_mock.py` **was** executed against a scripted stand-in model with the same message shapes. Verify against current provider docs before shipping.

## 1. Identity & problem
LLMs only emit text. **Tool calling** lets a model *request* actions — look up data, call an API, run code, query a database — by emitting a structured call that **your application executes**; the result is fed back and the model continues. It converts a text generator into a component that can fetch fresh facts, compute exactly, and act.

## 2. History
Toolformer (Schick et al. 2023: models learn when to call APIs), ReAct (Yao et al. 2022: interleave reasoning and acting), WebGPT (2021), plugins (2023) → provider **function-calling APIs** (2023) with JSON-schema tool definitions → parallel tool calls, strict schema adherence/structured outputs, server-hosted tools (search, code execution) → **Model Context Protocol (MCP)**, an open standard introduced by Anthropic (Nov 2024) for exposing tools/resources to models across applications → agent frameworks (Ch 6.10). Improvement chain: prompt-parsing hacks → schema-validated calls → standardized tool servers.

## 3. The protocol (provider-agnostic)
```
App → model:  messages + tool definitions (name, description, input JSON schema)
Model → app:  assistant turn containing tool_use block(s): {id, name, input}   [stop_reason = "tool_use"]
App:          validate input → execute the function → capture result or error
App → model:  user turn with tool_result block(s): {tool_use_id, content, is_error?}  (ALL results of a turn in ONE message)
Model:        continues; repeats until it answers (stop_reason = "end_turn")
```
Key facts: the model **never executes** anything; `tool_use_id` links result to call; parallel calls in one turn should be answered together; failures should be returned as error results so the model can recover; the loop needs a step cap.

## 4. Designing tools
| Principle | Why |
|---|---|
| Few, clear, non-overlapping tools; name and describe like documentation for a new teammate | Selection accuracy falls as tools multiply and overlap |
| Precise JSON schema: types, enums, required fields, `additionalProperties: false` | Fewer malformed calls; some APIs offer a `strict: true` mode that guarantees schema-valid arguments |
| Return concise, relevant, human-readable results (not raw 10k-token dumps); paginate | Context is a budget |
| Make errors informative ("city 'Pariss' not found; did you mean Paris?") | Enables self-correction |
| Idempotent/safe by default; separate read tools from write tools; confirm destructive actions | Limits blast radius |
| Many tools? Tool search / deferred loading, routing to sub-tools | Keeps the prompt small |
Tool choice: `auto` (model decides) is the common mode; some newer models reject forced `tool_choice` (`any`/specific tool) — steer via instructions, strict schemas or structured outputs instead; check the current model notes.

## 5. Executed demonstration (scripted model)
`agent_loop_mock.py` implements the loop with a deterministic fake model that (a) issues three parallel calls, (b) is "hijacked" by an instruction hidden in a tool result and requests a forbidden tool, (c) makes a malformed call, then (d) corrects itself and answers. Output:
```
(0, 'get_weather', 'ok')            (0, 'get_weather', 'ok')            (0, 'read_note', 'ok')
(1, 'delete_all_files', "error: tool 'delete_all_files' is not allowed")
(1, 'add', "error: arg 'a' must be number")
(2, 'add', 'ok')
results per user turn: [3, 2, 1]    # all parallel results returned together in one user message
final: Paris 18C rain, Rome 26C sun. The note mentions a 5pm meeting. 2+3=5. (I did not run the destructive tool.)
```
What it shows: **allow-listing and schema validation in code** blocked the injected destructive call and the bad argument type, errors were returned to the model as `is_error` results, and the step cap bounds runaway loops. Real models behave differently — this verifies your *harness*, not the model.

## 6. Provider SDK usage (templates, not executed)
Tool runner (the SDK runs the loop for you; Anthropic Python SDK, beta helper):
```python
import anthropic
from anthropic import beta_tool
client = anthropic.Anthropic()

@beta_tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location.

    Args:
        location: City and state, e.g., San Francisco, CA.
        unit: Temperature unit, either "celsius" or "fahrenheit".
    """
    return f"72°F and sunny in {location}"        # your implementation

runner = client.beta.messages.tool_runner(model="claude-opus-5-5", max_tokens=16000, tools=[get_weather],
                                          messages=[{"role": "user", "content": "What's the weather in Paris?"}])
for message in runner: print(message)
```
Manual loop (full control; same protocol as Section 3):
```python
messages = [{"role": "user", "content": user_input}]
while True:
    response = client.messages.create(model="claude-opus-5-5", max_tokens=16000, tools=tools, messages=messages)
    if response.stop_reason == "end_turn": break
    messages.append({"role": "assistant", "content": response.content})
    results = [{"type": "tool_result", "tool_use_id": b.id, "content": execute_tool(b.name, b.input)}
               for b in response.content if b.type == "tool_use"]
    messages.append({"role": "user", "content": results})
```
(Production code should also handle `pause_turn`, `max_tokens`, refusals, validation, timeouts and `is_error` results as in the executed harness.)
Server-hosted tools (web search, code execution, etc.) run on the provider's infrastructure and arrive as result blocks without your execution code. Always parse tool inputs as JSON, never by string matching.

## 7. Security checklist
Validate every argument in code · allow-list tool names · least-privilege credentials per tool · sandbox code execution · rate-limit and log every call · human approval for irreversible actions (payments, deletes, external messages) · treat tool *results* as untrusted data (prompt injection; see 6.3/6.11) · avoid giving the model secrets · defend against SSRF/path traversal/SQL injection in tool implementations (the model is an untrusted caller).

## 8. Evaluation
Tool-selection accuracy (right tool), argument accuracy (exact/semantic match), task success end-to-end, number of calls/latency/cost per task, invalid-call rate, recovery rate after errors, safety tests (injection, forbidden tools). Use recorded traces for regression tests; mock tools for determinism.

## 9. Common mistakes
Vague descriptions causing wrong tool choice · returning enormous outputs · swallowing exceptions instead of returning errors · splitting parallel results across messages · no step/budget cap · letting tool results contain instructions the model follows · assuming the model's JSON is valid without checking.

## 10. Case study (template)
Customer-support assistant with tools `lookup_order`, `refund` (requires approval), `search_kb`. Metrics: correct-tool rate, refund policy violations (must be 0), resolution rate, average calls, p95 latency. Start with workflow-style routing; add autonomy only if evals show benefit (Ch 6.10).

## Interview Q&A
- *Who executes the function?* Your application; the model only proposes a call.
- *Why return errors to the model?* It can retry with corrected arguments.
- *Parallel tool calls?* Independent calls in one turn; return all results together.
- *How do you make tool calling reliable?* Clear schemas, strict mode/validation, few well-described tools, evals, retries.
- *What is MCP?* An open protocol for exposing tools/resources/prompts to AI applications through standardized servers.

## Exercises
Extend `agent_loop_mock.py` with a rate limiter and an approval gate for a `send_email` tool; write a schema for a SQL-query tool with a read-only allow-list; build an eval of 30 tool-selection cases.

## Further reading
Yao et al. (2022) ReAct; Schick et al. (2023) Toolformer; Model Context Protocol specification (modelcontextprotocol.io); provider tool-use documentation (Anthropic, OpenAI); OWASP LLM Top 10 (excessive agency, insecure tool use).
