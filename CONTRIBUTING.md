# Contributing to the Rebuno Python SDK

Thanks for your interest in contributing. This guide covers how to set up the project locally and submit changes.

## Prerequisites

- **Python** 3.11+ (CI runs 3.11 through 3.14)
- **[uv](https://docs.astral.sh/uv/)**

## Getting Started

```bash
uv sync                  # install dependencies
uv run pytest            # run the tests
uv run ruff check        # lint
uv run ruff format       # format
```

SDK documentation lives in the main repo under
[docs/sdk/python](https://github.com/rebuno/rebuno/tree/main/docs/sdk/python).
If you change public API surface or behavior, update it there.

## Submitting Changes

1. Fork the repo and create a branch from `main`.
2. Make your changes. Add tests for new functionality.
3. Run `uv run ruff format`, then make sure `uv run pytest` and `uv run ruff check` pass — CI runs those two plus `ruff format --check`.
4. Open a pull request with a clear description of what changed and why.

## Reporting Issues

Open an issue on GitHub. Include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Relevant logs or error messages

## License

By submitting a contribution, you agree that it is licensed under the
[MIT License](LICENSE), the same terms that cover the rest of the project.
