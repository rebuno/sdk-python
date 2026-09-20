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

- [Getting started](https://docs.rebuno.io/sdk/python/getting-started): install, configuration, the dispatch loop, and a complete example.
- [Agents](https://docs.rebuno.io/sdk/python/agents): the `Agent` host, input binding, `run` vs `app`, dispatch and resume, lifecycle.
- [Tools](https://docs.rebuno.io/sdk/python/tools): `@tool`, `wrap_tool`, idempotency, blocking work, and wrapping MCP tools.
- [LLM calls](https://docs.rebuno.io/sdk/python/llm-calls): `http_client()` and `RebunoTransport`.
- [Local steps](https://docs.rebuno.io/sdk/python/steps): `rebuno.step()` for durable local work.
- [Clients](https://docs.rebuno.io/sdk/python/client): creating and inspecting executions, and approvals.
- [Errors](https://docs.rebuno.io/sdk/python/errors): the exception hierarchy.
- [How it works](https://docs.rebuno.io/sdk/python/internals): step identity, replay, heartbeats, and the kernel protocol.

## License

[MIT](LICENSE)
