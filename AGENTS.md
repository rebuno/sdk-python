# Working on the Rebuno Python SDK

This repository contains the async Python SDK for Rebuno. It hosts agents,
routes effects through the kernel, and exposes a client for executions and
approvals. The Go kernel owns durable state, policy decisions, and step identity.
Read [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/rebuno/agent.py` | FastAPI webhook host, dispatch tasks, input binding, shutdown. |
| `src/rebuno/execution.py` | Execution context, step decisions, replay, lease heartbeats. |
| `src/rebuno/_kernel.py`, `src/rebuno/client.py` | Signed agent requests and the public client API. |
| `src/rebuno/tool.py`, `src/rebuno/step.py`, `src/rebuno/mcp.py` | Durable tools, local steps, and MCP adapters. |
| `src/rebuno/http_client.py` | LLM HTTP interception, response recording, streaming, refusal responses. |
| `src/rebuno/types.py`, `src/rebuno/errors.py` | Wire models, SDK errors, provider refusal conversion. |
| `src/rebuno/_internal/inputs.py` | Handler signature inspection and input validation. |
| `src/rebuno/__init__.py`, `src/rebuno/py.typed` | Public exports and typed-package marker. |
| `tests/` | Pytest coverage; shared execution fakes in `tests/conftest.py`. |

SDK documentation and examples live in the main Rebuno repository under
`docs/sdk/python/` and `examples/python/`. The TypeScript SDK and dashboard are
separate repositories. Edit source under `src/`; build output and environments
such as `dist/` and `.venv/` are not source.

## Development and validation

Use Python 3.11+ and uv. The supported interpreter matrix is in
[.github/workflows/ci.yml](.github/workflows/ci.yml). Run commands from this root:

| Command | Purpose |
| --- | --- |
| `uv sync --frozen` | Install the locked environment, including development dependencies. |
| `uv run pytest tests/test_execution.py` | Example of focused tests; adjust to the affected files. |
| `make test` | Run the full pytest suite. |
| `make lint` | Run Ruff lint and formatting checks. |
| `make format` | Apply Ruff formatting and lint fixes. |
| `uv build` | Build the wheel and source distribution when packaging changes. |

For Python changes, run focused tests while iterating, then `make test` and
`make lint` before handing off. Scope formatting to changed files when unrelated
edits are present. Keep `uv.lock` consistent with dependency changes in
`pyproject.toml`; CI uses a frozen install.

Tests use fake kernels, `httpx2.MockTransport`, and in-process ASGI requests.
Prefer those seams for unit tests; a running kernel or live LLM provider is not
required for the existing suite. Pytest uses automatic asyncio mode. Restore
execution context tokens and clean up tasks, clients, and streams in tests.
For documentation-only changes, check paths, commands, and consistency with the
implementation; runtime tests are unnecessary. Report any checks that could not
run and why.

## SDK invariants

- Submit effects to the kernel before invoking their bodies. The kernel returns
  the step ID and counts occurrences; keep that responsibility in the kernel.
  Replay returns the recorded result or error without invoking the effect.
  Preserve the `safe_to_retry` and `at_most_once` idempotency contracts.
- Keep execution state scoped to each dispatch with `ContextVar`. Reset it on
  every exit path and preserve the owning event loop for kernel-client I/O.
  Blocking synchronous work must not starve lease heartbeats.
- Sign the exact request bytes sent to the kernel. Verify webhook signatures
  against the raw body before decoding it. Preserve the dispatch ID and attempt
  headers on durable mutations and heartbeats.
- Deduplicate delivery attempts within their dispatch. A superseding delivery
  must stop the previous handler's writes without waiting indefinitely for that
  handler. Cancel and await heartbeat tasks when their dispatch ends.
- Preserve `Blocked`, `Terminated`, and `LeaseSuperseded` as control-flow signals.
  A suspended or superseded handler must not complete or fail the execution.
  Keep policy refusals, rate limits, tool failures, and transport errors distinct
  through tool wrappers and provider-error conversion.
- Tools and local steps require an active execution. LLM HTTP interception
  passes requests through when there is no execution context or eligible JSON
  body. Preserve those boundaries.
- Preserve callable signatures, defaults, metadata, and JSON-recorded arguments
  when changing tool or MCP wrappers; frameworks inspect them to build schemas.
- Keep streamed deltas best-effort and the recorded response durable. Cover
  stream completion, early consumer close, midstream failure, and UTF-8 chunk
  boundaries when changing the transport. Replay must reconstruct the recorded
  HTTP status, content type, and body without calling the provider.

## Public API and documentation

Keep public exports, type annotations, Pydantic models, and error mappings in
sync with behavior changes. Preserve Python 3.11 compatibility and the async
API. Reuse the existing `httpx2` transport and callable adapter seams.

Add focused regression coverage for affected behavior, including replay,
suspension, stale leases, or concurrent executions where relevant. Check the
kernel's `/v0` protocol and the TypeScript SDK when changing shared semantics;
flag any coordinated changes they need.

Update the [Python SDK documentation](https://github.com/rebuno/rebuno/tree/main/docs/sdk/python)
and affected examples in the main repository when public API or behavior changes.
In a sibling checkout these are under `../rebuno/docs/sdk/python/` and
`../rebuno/examples/python/`. Keep the README example and relevant guidance under
`../rebuno/skills/rebuno/references/` consistent with those changes.

## Comments, tests, and documentation style

Write repository content for someone reading the finished system with no access
to the task discussion. Changes should read as a natural part of the codebase.

- Keep comments and docstrings concise. Explain non-obvious intent, invariants,
  or constraints when the code cannot express them clearly. Omit comments that
  restate the code or announce an edit.
- Keep conversation references, review replies, task instructions, and abandoned
  approaches out of code, tests, and documentation. Put implementation history
  and change rationale in PR descriptions or commit messages.
- Describe behavior directly in the present tense. Avoid change-relative wording
  such as "now", "new", "previously", "we changed", or "X instead of Y" when it
  only makes sense in the context of the change. Explain a comparison only when
  it helps the reader understand a lasting distinction or compatibility rule.
- Update existing documentation and examples in place. Integrate the final
  behavior into the relevant section; avoid appended fix notes, repeated caveats,
  and explanations of superseded designs. Release notes and migration guides
  can describe changes over time when that is their purpose.
- Name tests for the behavior or invariant they verify. Keep assertions focused
  on meaningful outcomes and failure modes. Preserve useful regression coverage;
  avoid redundant tests, assertions that merely mirror implementation details,
  and test commentary that recounts the debugging session.
- Review the diff for wording that depends on knowing the conversation or the
  previous patch. Remove it or rewrite it as a standalone explanation of the
  current system.
