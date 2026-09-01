# Rebuno Python SDK

Python SDK for [Rebuno](https://github.com/rebuno/rebuno), an open-source
execution runtime for production agents.

## Install

```bash
pip install rebuno
```

Requires Python 3.11 or later.

## An agent

```python
from rebuno import Agent, tool


@tool
async def search(query: str) -> list[str]:
    return [f"result for {query}"]


async def process(prompt: str) -> dict:
    hits = await search(prompt)
    return {"answer": hits}


agent = Agent("dev-agent", secret="dev-secret", base_url="http://localhost:8080")
agent.run(process, port=5000)
```

Every effect goes to the kernel as a step before it runs. On a re-dispatch the
handler runs again from the top, and any step with a recorded result replays it
instead of running a second time.

## Documentation

- [Getting started](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/getting-started.md): install, configuration, the dispatch loop, and a complete example.
- [Agents](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/agents.md): the `Agent` host, input binding, `run` vs `app`, dispatch and resume, lifecycle.
- [Tools](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/tools.md): `@tool`, `wrap_tool`, idempotency, blocking work, and wrapping MCP tools.
- [LLM calls](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/llm-calls.md): `http_client()` and `RebunoTransport`.
- [Local steps](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/steps.md): `rebuno.step()` for durable local work.
- [Clients](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/client.md): creating and inspecting executions, and approvals.
- [Errors](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/errors.md): the exception hierarchy.
- [How it works](https://github.com/rebuno/rebuno/blob/main/docs/sdk/python/internals.md): step identity, replay, heartbeats, and the kernel protocol.

## License

[MIT](LICENSE)
