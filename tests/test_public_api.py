import rebuno


def test_public_surface():
    for name in ("Agent", "Client", "tool", "step", "execution", "types"):
        assert hasattr(rebuno, name), name
    from rebuno import (  # noqa: F401
        APIError,
        Blocked,
        PolicyError,
        RebunoError,
        Terminated,
        ToolError,
    )
