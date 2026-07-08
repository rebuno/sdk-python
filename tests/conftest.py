import pytest

pytest_plugins: list[str] = []


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
