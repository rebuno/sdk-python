# Working on the Rebuno Python SDK

This is Rebuno's async Python SDK. The Go kernel owns durable state, policy
decisions, and step identity. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup
and the contribution workflow.

## Development and validation

Use uv and the Python versions declared in `pyproject.toml` and CI. Install with
`uv sync --frozen`; for Python changes, run focused tests, then `make test` and
`make lint`. The [Makefile](Makefile) defines formatting and validation commands.
Keep the lockfile consistent with dependency changes.

Reuse fake kernels and transport mocks for unit tests. Restore execution context
and clean up tasks, clients, and streams. For documentation-only changes, check
paths, commands, and implementation consistency; runtime tests are unnecessary.
Report checks that could not run and why.

## SDK constraints

- Submit effects before invoking their bodies. The kernel assigns step IDs and
  occurrences; replay returns recorded outcomes without invoking the effect.
  Preserve the `safe_to_retry` and `at_most_once` contracts.
- Scope execution context to each dispatch and preserve the owning event loop
  for kernel I/O. Keep heartbeats responsive and clean them up on every exit.
  Superseding a handler must stop its writes without waiting indefinitely for it.
- Sign exact request bytes, verify raw webhook bodies, and preserve dispatch ID
  and attempt headers. Attempt ordering is scoped to a dispatch ID.
- Keep suspension, termination, and lease loss distinct from effect failures.
  Suspended or superseded handlers must not complete or fail the execution,
  including when user code catches a control-flow exception.
- Preserve framework-facing callable signatures, defaults, metadata, and recorded
  arguments. Keep public exports, type annotations, and error mappings aligned
  with behavior changes and the supported Python versions.
- Keep stream deltas best-effort and response recording durable. Test replay,
  early consumer close, and midstream failure when changing interception.

Check the kernel protocol and TypeScript SDK when changing shared semantics.
Update the [Python SDK docs](https://github.com/rebuno/rebuno/tree/main/docs/sdk/python),
examples, and relevant agent-building guidance in the main Rebuno repository
with public behavior changes. These are available in `../rebuno/` when using
sibling checkouts.

## Comments, tests, and documentation style

Write for someone reading the finished system with no access to the task
conversation. Changes should read as a natural part of the codebase.

- Keep comments and docstrings sparse and concise. Explain non-obvious intent,
  invariants, or constraints; omit restatements of code and announcements of edits.
- Keep conversation references, review replies, and abandoned approaches out of
  source, tests, and docs. Put change history in PRs, commits, release notes, or
  migration guides when relevant.
- Describe current behavior directly in the present tense. Avoid change-relative
  wording such as "now", "previously", or "X instead of Y" unless it explains a
  lasting distinction or compatibility rule.
- Update existing documentation and examples in place. Avoid appended fix notes
  and repeated caveats. Review additions for wording that depends on the task
  discussion or previous patch.
- Add focused regression tests for meaningful behavior and failure modes. Name
  tests for the behavior they verify; avoid redundant coverage, assertions that
  mirror implementation details, and commentary about the debugging session.
