# Rebuno Python SDK

Python client library for the [Rebuno](https://github.com/rebuno/rebuno) agent execution runtime.

## Installation

```bash
pip install rebuno
```

## Quick Start

```python
from rebuno import Client

client = Client(base_url="http://localhost:8080")

execution = await client.create("my-agent", input={"prompt": "hello"})
print(execution.id)
```

`Client` defaults to `REBUNO_URL` and `REBUNO_API_KEY` env vars when no args are passed.

## Building an Agent

```python
from rebuno import Agent, execution

agent = Agent("my-agent")

async def process(prompt: str) -> dict:
    print("running execution", execution.id)
    return {"answer": prompt.upper()}

agent.run(process)
```

The handler signature is the input schema — `process(prompt: str)` makes `prompt` a required field. Pass a `pydantic.BaseModel` parameter for validation, or `input: dict` for the raw claim. Use `from rebuno import execution` to access the current execution's id, session_id, history, etc.

## Tools

```python
from rebuno import tool

@tool("web.search")
async def search(query: str, limit: int = 10) -> list[str]:
    return [...]
```

Tools are plain module-level functions. The wrapper submits an intent to the kernel before running the body. Hand them to your framework as a list:

```python
graph = create_agent(llm, [search, ...])
```

## Building a Runner

```python
from rebuno import Runner, tool

@tool("compute.heavy")
async def heavy(data: str) -> str:
    return process(data)

Runner("compute-1").run()
```

The runner advertises every `@tool` it imports, publishes their schemas to the kernel, and services job assignments. `@tool(remote=True)` lets you declare a stub in agent code for type-checked imperative calls.

## Local MCP

The agent connects to the MCP server directly. Install with `pip install rebuno[mcp]`:

```python
from rebuno import MCPServer

github = MCPServer(
    "github",
    url="https://api.githubcopilot.com/mcp/",
    headers={"Authorization": "Bearer xxx"},
)

# Inside your handler:
graph = create_agent(llm, [..., *github.tools])
```

## Remote tools (incl. remote MCP)

Discover tools that runners host elsewhere — the agent never holds credentials, never opens MCP transport. Schemas come from the kernel directory.

```python
from rebuno import remote

github  = remote.Tools("github")
compute = remote.Tools("compute")

# Inside your handler:
graph = create_agent(llm, [..., *github.tools, *compute.tools])
```

Each call routes through the kernel to whichever runner advertises that tool ID. Works for any source — `@tool` Python functions on a runner, MCP servers hosted by a runner, or both.

## Documentation

See the [full documentation](https://github.com/rebuno/rebuno/tree/main/docs).

## License

MIT
