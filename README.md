# Rebuno Python SDK

Rebuno gives your agents durable execution (crash and resume without re-running side effects), an event-sourced record of everything they did, and optional governance over what they're allowed to do.

Durability works by recording every non-deterministic effect your handler
produces, so a resumed run replays the recorded result instead of doing the work
again. The SDK gives you three ways to record an effect:

- `@rebuno.tool` — mark an async function as a durable tool call
- `rebuno.http_client()` — an `httpx` client that records LLM calls as durable
  steps (drop it into your OpenAI/Anthropic client)
- `rebuno.step()` — record non-deterministic local work (time, randomness, ids) so it
  replays identically

## Installation

```bash
pip install rebuno
```

## Quick start (create an execution)

```python
from rebuno import Client

client = Client(base_url="http://localhost:8080")
execution = await client.create("dev-agent", input={"prompt": "hello"})
print(execution.id)
```

`Client` falls back to the `REBUNO_URL` and `REBUNO_API_KEY` environment variables
when `base_url` / `api_key` are omitted.

## Building an agent

```python
from rebuno import Agent, tool


@tool
async def search(query: str) -> list[str]:
    return [f"result for {query}"]


async def process(prompt: str) -> dict:
    hits = await search(prompt)
    return {"answer": hits}


agent = Agent("dev-agent", secret="dev-secret", kernel_url="http://localhost:8080")
agent.run(process, port=5000)
```

The handler's signature *is* the input schema: `process(prompt: str)` makes `prompt`
a required input field. A single `pydantic.BaseModel` parameter gets validated input;
a single `input: dict` (or `input: Any`) parameter receives the raw input unchanged.

`agent.run(...)` binds `process` and serves the webhook endpoint with uvicorn
(this call blocks). The kernel calls that webhook to dispatch each execution; the
agent runs `process` and records every tool call durably, so crashes and pending
approvals resume transparently on the next dispatch — your handler code doesn't need
to know the difference.

If you want to mount the agent into an existing service, or serve it with your own
uvicorn/gunicorn process, use `agent.app` (a FastAPI app) instead of `agent.run(...)`.

## Tools

```python
from rebuno import tool


@tool
async def search(query: str, limit: int = 10) -> list[str]:
    ...
```

Tools are plain async functions — hand them to your framework as a list:

```python
agent = create_agent(llm, [search, ...])
```

Use `@tool("custom_id")` to set an explicit tool id, and
`@tool(idempotency="at_most_once")` for destructive operations that must not be
retried automatically (e.g. sending an email or charging a card).

If a tool does blocking or CPU-bound work, offload it so it doesn't block the event
loop:

```python
@tool
async def render(doc: str) -> bytes:
    return await asyncio.to_thread(render_sync, doc)
```

## Durable LLM calls

LLM calls are the most expensive and least deterministic thing an agent does, so
Rebuno records them too — without you rewriting how you call the model.
`http_client()` returns an `httpx.AsyncClient` you hand to your provider's async
client:

```python
from openai import AsyncOpenAI
import rebuno

llm = AsyncOpenAI(http_client=rebuno.http_client())
```

It works as an httpx transport that sits under the provider SDK: on the first
run it forwards the request to the provider and records the response as a durable
step (`kind=llm_call`, the same machinery as tool calls); on resume it replays
the recorded response instead of calling — and paying for — the model again. The
request's model field is used as the step target; pass `model_field=...` if your
provider names it differently. Extra kwargs (e.g. `timeout`) are forwarded to
`httpx.AsyncClient`.

Recording only happens inside an execution — outside one, the client is a plain
passthrough. Two current limits: streaming responses (`stream=True`) are passed
through un-recorded (you'll get a warning), and non-JSON request bodies aren't
recognized as LLM calls.

## Durable local work — `rebuno.step()`

Wrap non-deterministic local computation — the current time, random choices, fresh
ids — so its result is recorded once and replays identically on resume:

```python
from rebuno import step

chosen = await step("pick_winner", random.choice, args={"candidates": candidates})
```

## Human-in-the-loop / approvals

Approvals are inspected and resolved through `Client`:

```python
pending = await client.list_approvals()
await client.grant_approval(pending[0].id, decided_by="alice")
```

## License

MIT
