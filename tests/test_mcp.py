import inspect
from types import SimpleNamespace

from conftest import FakeKernel, install_context

from rebuno.execution import _reset_current
from rebuno.mcp import wrap_mcp_tool, wrap_mcp_tools
from rebuno.types import StepDecision


def descriptor(
    name="get_weather",
    description="Get weather",
    required=("city",),
    optional=("units",),
):
    props = {p: {"type": "string"} for p in (*required, *optional)}
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": props, "required": list(required)},
    )


def make_call(record):
    async def call(name, args):
        record.append((name, dict(args)))
        return {"ok": True, "echo": args}

    return call


async def test_wrap_routes_through_invoke_tool_with_prefix():
    k = FakeKernel(StepDecision(decision="proceed"))
    record: list = []
    fn = wrap_mcp_tool(descriptor(), call=make_call(record), prefix="weather")
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert k.captured["kind"] == "tool_call"
    assert k.captured["target"] == "weather_get_weather"
    assert k.captured["args"] == {"city": "London"}
    assert record == [("get_weather", {"city": "London"})]
    assert out == {"ok": True, "echo": {"city": "London"}}
    assert k.completed == [{"ok": True, "echo": {"city": "London"}}]


async def test_wrap_without_prefix_uses_bare_name():
    k = FakeKernel(StepDecision(decision="proceed"))
    fn = wrap_mcp_tool(descriptor(), call=make_call([]))
    token = install_context(k)
    try:
        await fn(city="Paris")
    finally:
        _reset_current(token)
    assert k.captured["target"] == "get_weather"


async def test_wrap_strips_none_valued_args():
    k = FakeKernel(StepDecision(decision="proceed"))
    record: list = []
    fn = wrap_mcp_tool(descriptor(), call=make_call(record), prefix="w")
    token = install_context(k)
    try:
        await fn(city="Rome", units=None)
    finally:
        _reset_current(token)
    assert k.captured["args"] == {"city": "Rome"}
    assert record == [("get_weather", {"city": "Rome"})]


async def test_wrap_forwards_idempotency():
    k = FakeKernel(StepDecision(decision="proceed"))
    fn = wrap_mcp_tool(descriptor(), call=make_call([]), idempotency="at_most_once")
    token = install_context(k)
    try:
        await fn(city="Oslo")
    finally:
        _reset_current(token)
    assert k.captured["idempotency"] == "at_most_once"


async def test_wrap_replays_recorded_result_without_calling_mcp():
    k = FakeKernel(StepDecision(decision="replay", result={"cached": True}))
    record: list = []
    fn = wrap_mcp_tool(descriptor(), call=make_call(record), prefix="w")
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert out == {"cached": True}
    assert record == []


async def test_synthetic_signature_and_introspection():
    fn = wrap_mcp_tool(descriptor(), call=make_call([]), prefix="w")
    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["city", "units"]
    city, units = sig.parameters["city"], sig.parameters["units"]
    assert city.kind is inspect.Parameter.KEYWORD_ONLY
    assert city.default is inspect.Parameter.empty
    assert units.default is None
    assert fn.__name__ == "w_get_weather"
    assert fn.__doc__ == "Get weather"
    assert fn.__input_schema__["properties"]["city"] == {"type": "string"}


async def test_wrap_accepts_dict_descriptor():
    k = FakeKernel(StepDecision(decision="proceed"))
    desc = {
        "name": "search",
        "description": "Search",
        "inputSchema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    }
    fn = wrap_mcp_tool(desc, call=make_call([]), prefix="db")
    token = install_context(k)
    try:
        await fn(q="hi")
    finally:
        _reset_current(token)
    assert k.captured["target"] == "db_search"
    assert fn.__name__ == "db_search"


async def test_default_flatten_prefers_structured_content():
    k = FakeKernel(StepDecision(decision="proceed"))

    async def call(name, args):
        return SimpleNamespace(
            structured_content={"temp": 12}, content=[], data=object()
        )

    fn = wrap_mcp_tool(descriptor(), call=call, prefix="w")
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert out == {"temp": 12}
    assert k.completed == [{"temp": 12}]


async def test_default_flatten_handles_official_sdk_camelcase():
    k = FakeKernel(StepDecision(decision="proceed"))

    async def call(name, args):
        return SimpleNamespace(structuredContent={"temp": 9}, content=[])

    fn = wrap_mcp_tool(descriptor(), call=call)
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert out == {"temp": 9}


async def test_default_flatten_joins_text_content_blocks():
    k = FakeKernel(StepDecision(decision="proceed"))

    async def call(name, args):
        blocks = [
            SimpleNamespace(type="text", text="line one"),
            SimpleNamespace(type="text", text="line two"),
        ]
        return SimpleNamespace(structured_content=None, content=blocks)

    fn = wrap_mcp_tool(descriptor(), call=call)
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert out == "line one\nline two"


async def test_default_flatten_single_text_block_unwrapped():
    k = FakeKernel(StepDecision(decision="proceed"))

    async def call(name, args):
        return SimpleNamespace(
            structured_content=None,
            content=[SimpleNamespace(type="text", text="just one")],
        )

    fn = wrap_mcp_tool(descriptor(), call=call)
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert out == "just one"


async def test_default_flatten_passes_through_plain_value():
    k = FakeKernel(StepDecision(decision="proceed"))

    async def call(name, args):
        return {"already": "json"}

    fn = wrap_mcp_tool(descriptor(), call=call)
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert out == {"already": "json"}


async def test_to_result_override_applied_before_recording():
    k = FakeKernel(StepDecision(decision="proceed"))

    async def call(name, args):
        return SimpleNamespace(data={"hydrated": 1}, structured_content={"raw": 1})

    fn = wrap_mcp_tool(descriptor(), call=call, to_result=lambda r: r.data)
    token = install_context(k)
    try:
        out = await fn(city="London")
    finally:
        _reset_current(token)
    assert out == {"hydrated": 1}
    assert k.completed == [{"hydrated": 1}]


async def test_wrap_mcp_tools_wraps_each_descriptor():
    descs = [
        descriptor(name="a", required=("x",), optional=()),
        descriptor(name="b", required=(), optional=("y",)),
    ]
    fns = wrap_mcp_tools(descs, call=make_call([]), prefix="srv")
    assert [f.__name__ for f in fns] == ["srv_a", "srv_b"]
