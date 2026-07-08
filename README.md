# Rebuno Python SDK

Durable execution for agents, on any framework. Rebuno gives your agent crash-safe,
resumable runs without changing how you build it: write a normal async handler, wrap
the side effects you want recorded, and the kernel takes care of retries, replay, and
human-in-the-loop approvals.

The SDK is built around three primitives:

- `@rebuno.tool` — mark an async function as a durable tool call
- `rebuno.Agent(agent_id).run(process)` — receive webhook dispatches from the kernel
  and run your handler per execution
- `rebuno.step()` — record non-deterministic local work (time, randomness, ids) so it
  replays identically

The SDK is async-only.

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
