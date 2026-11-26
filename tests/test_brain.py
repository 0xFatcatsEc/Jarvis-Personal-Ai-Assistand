"""Tests for brain.py: adapters, fallback, retries and prompt assembly."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from brain import (
    AllProvidersFailed,
    Brain,
    Completion,
    LLMError,
    Message,
    TextDelta,
    ToolCall,
    build_system_prompt,
    load_system_template,
)
from config import LlmConfig, ProviderConfig, RoutingConfig


async def no_sleep(_delay: float) -> None:
    return None


def ollama(**kw) -> ProviderConfig:
    kw.setdefault("model", "llama3")
    kw.setdefault("base_url", "http://ollama.test")
    return ProviderConfig(name="ollama", max_retries=kw.pop("max_retries", 0), **kw)


def openai(**kw) -> ProviderConfig:
    kw.setdefault("model", "gpt-4o-mini")
    kw.setdefault("base_url", "http://openai.test/v1")
    kw.setdefault("api_key", "sk-test")
    return ProviderConfig(name="openai", max_retries=kw.pop("max_retries", 0), **kw)


def anthropic(**kw) -> ProviderConfig:
    kw.setdefault("model", "claude-haiku")
    kw.setdefault("base_url", "http://anthropic.test/v1")
    kw.setdefault("api_key", "ak-test")
    return ProviderConfig(name="anthropic", max_retries=kw.pop("max_retries", 0), **kw)


def make_brain(providers, handler, *, routing=None, sleep=no_sleep, backends=None) -> Brain:
    cfg = LlmConfig(providers=providers, routing=routing or RoutingConfig(), temperature=0.2, max_tokens=64)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Brain(cfg, client=client, sleep=sleep, backends=backends)


async def run_stream(brain: Brain, messages=None, tools=None):
    messages = messages or [Message(role="user", content="hi")]
    events = [event async for event in brain.stream(messages, tools=tools)]
    completion = events[-1]
    assert isinstance(completion, Completion)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    return text, completion


OLLAMA_BODY = "\n".join(
    [
        json.dumps({"message": {"role": "assistant", "content": "Hello"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": " there"}, "done": False}),
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "weather"}}}],
                },
                "done": True,
                "done_reason": "stop",
            }
        ),
    ]
)


def test_ollama_streams_text_and_tool_call():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=OLLAMA_BODY)

    brain = make_brain([ollama()], handler)
    text, completion = asyncio.run(run_stream(brain, tools=[{"name": "web_search", "description": "d", "parameters": {}}]))

    assert text == "Hello there"
    assert completion.provider == "ollama" and completion.text == "Hello there"
    assert completion.tool_calls == [ToolCall(id="ollama-0", name="web_search", arguments={"query": "weather"})]
    assert completion.finish_reason == "stop"
    assert seen["url"] == "http://ollama.test/api/chat"
    assert seen["body"]["model"] == "llama3"
    assert seen["body"]["options"]["temperature"] == 0.2
    assert seen["body"]["options"]["num_predict"] == 64
    assert seen["body"]["tools"][0]["function"]["name"] == "web_search"


OLLAMA_THINKING_BODY = "\n".join(
    [
        json.dumps({"message": {"role": "assistant", "thinking": "The user said hi. "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "thinking": "Answer briefly."}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "Hi "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "there"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}),
    ]
)


def test_ollama_keeps_reasoning_out_of_the_answer():
    brain = make_brain([ollama()], lambda request: httpx.Response(200, text=OLLAMA_THINKING_BODY))

    text, completion = asyncio.run(run_stream(brain))

    assert text == "Hi there"
    assert completion.text == "Hi there"


def test_ollama_reasoning_without_answer_reports_budget():
    body = "\n".join(
        [
            json.dumps({"message": {"role": "assistant", "thinking": "thinking..."}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length"}),
        ]
    )
    brain = make_brain([ollama()], lambda request: httpx.Response(200, text=body))

    with pytest.raises(LLMError, match="reasoning but no answer"):
        asyncio.run(run_stream(brain))


def test_ollama_converts_tool_history():
    def handler(request):
        body = json.loads(request.content)
        assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "notes"
        assert body["messages"][3] == {"role": "tool", "content": "saved", "tool_name": "notes"}
        return httpx.Response(200, text=OLLAMA_BODY)

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="note this"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="notes", arguments={"text": "x"})]),
        Message(role="tool", content="saved", tool_call_id="c1", name="notes"),
    ]
    brain = make_brain([ollama()], handler)
    asyncio.run(run_stream(brain, messages))


OPENAI_BODY = "\n\n".join(
    [
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"web_search","arguments":"{\\"query\\""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"news\\"}"}}]},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
)


def test_openai_streams_and_assembles_tool_arguments():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text=OPENAI_BODY)

    brain = make_brain([openai()], handler)
    text, completion = asyncio.run(run_stream(brain, tools=[{"name": "web_search", "description": "d", "parameters": {}}]))

    assert text == "Hi"
    assert completion.tool_calls == [
        ToolCall(id="call_1", name="web_search", arguments={"query": "news"})
    ]
    assert completion.finish_reason == "tool_calls"
    assert seen["url"] == "http://openai.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["tools"][0]["type"] == "function"
    assert seen["body"]["max_tokens"] == 64


def test_openai_history_shape():
    def handler(request):
        body = json.loads(request.content)
        assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"text": "x"}'
        assert body["messages"][2] == {"role": "tool", "content": "saved", "tool_call_id": "c1"}
        return httpx.Response(200, text=OPENAI_BODY)

    messages = [
        Message(role="system", content="sys"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="notes", arguments={"text": "x"})]),
        Message(role="tool", content="saved", tool_call_id="c1", name="notes"),
    ]
    brain = make_brain([openai()], handler)
    asyncio.run(run_stream(brain, messages))


ANTHROPIC_BODY = "\n\n".join(
    [
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Sunny"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lights"}}',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"room\\": \\"kitchen\\"}"}}',
        'data: {"type":"content_block_stop","index":1}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
        "data: [DONE]",
    ]
)


def test_anthropic_streams_text_and_tool_input():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        return httpx.Response(200, text=ANTHROPIC_BODY)

    messages = [
        Message(role="system", content="sys one"),
        Message(role="system", content="sys two"),
        Message(role="user", content="lights?"),
    ]
    brain = make_brain([anthropic()], handler)
    text, completion = asyncio.run(run_stream(brain, messages, tools=[{"name": "lights", "description": "d", "parameters": {"type": "object"}}]))

    assert text == "Sunny"
    assert completion.tool_calls == [ToolCall(id="toolu_1", name="lights", arguments={"room": "kitchen"})]
    assert completion.finish_reason == "tool_use"
    assert seen["url"] == "http://anthropic.test/v1/messages"
    assert seen["key"] == "ak-test"
    assert seen["version"] == "2023-06-01"
    assert seen["body"]["system"] == "sys one\n\nsys two"
    assert seen["body"]["tools"][0]["input_schema"] == {"type": "object"}


def test_anthropic_groups_tool_results_into_one_user_turn():
    def handler(request):
        body = json.loads(request.content)
        convo = body["messages"]
        tool_turn = convo[-1]
        assert tool_turn["role"] == "user"
        assert [b["type"] for b in tool_turn["content"]] == ["tool_result", "tool_result"]
        return httpx.Response(200, text=ANTHROPIC_BODY)

    messages = [
        Message(role="user", content="do two things"),
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="a", name="notes", arguments={}),
            ToolCall(id="b", name="notes", arguments={}),
        ]),
        Message(role="tool", content="ok a", tool_call_id="a"),
        Message(role="tool", content="ok b", tool_call_id="b"),
    ]
    brain = make_brain([anthropic()], handler)
    asyncio.run(run_stream(brain, messages))


def test_fallback_moves_to_next_provider_on_server_error():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/chat":
            return httpx.Response(503, text="down")
        return httpx.Response(200, text=OPENAI_BODY)

    brain = make_brain([ollama(max_retries=0), openai()], handler)
    text, completion = asyncio.run(run_stream(brain))
    assert text == "Hi"
    assert completion.provider == "openai"
    assert calls == ["/api/chat", "/v1/chat/completions"]


def test_auth_failure_does_not_fall_back():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(401, text="bad key")

    brain = make_brain([ollama(), openai()], handler)
    with pytest.raises(LLMError) as info:
        asyncio.run(run_stream(brain))
    assert info.value.reason == "auth"
    assert calls == ["/api/chat"]


def test_retries_transient_failures_then_succeeds():
    calls = []
    delays = []

    async def sleep(delay):
        delays.append(delay)

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, text=OLLAMA_BODY)

    brain = make_brain([ollama(max_retries=2)], handler, sleep=sleep)
    text, completion = asyncio.run(run_stream(brain))
    assert text == "Hello there"
    assert completion.provider == "ollama"
    assert len(calls) == 3
    assert delays == [0.4, 0.8]


def test_no_fallback_after_tokens_were_emitted():
    calls = []

    async def flaky(provider, messages, tools, max_tokens, temperature, client):
        yield TextDelta("partial")
        raise LLMError("connection", "dropped", "ollama")

    async def never(provider, messages, tools, max_tokens, temperature, client):
        calls.append("openai")
        yield TextDelta("should not run")

    brain = make_brain(
        [ollama(), openai()],
        lambda request: httpx.Response(500, text="unused"),
        backends={"ollama": flaky, "openai": never},
    )
    with pytest.raises(LLMError, match="connection"):
        asyncio.run(run_stream(brain))
    assert calls == []


def test_all_failures_are_reported():
    brain = make_brain([ollama(), openai()], lambda request: httpx.Response(500, text="down"))
    with pytest.raises(AllProvidersFailed) as info:
        asyncio.run(run_stream(brain))
    assert [f.provider for f in info.value.failures] == ["ollama", "openai"]


def test_local_only_skips_cloud_providers():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(200, text=OLLAMA_BODY)

    brain = make_brain([ollama(), openai()], handler, routing=RoutingConfig(mode="local_only"))
    _, completion = asyncio.run(run_stream(brain))
    assert completion.provider == "ollama"
    assert calls == ["ollama.test"]


def test_no_usable_provider_raises():
    brain = make_brain([openai(api_key=None)], lambda request: httpx.Response(200, text=OPENAI_BODY))
    with pytest.raises(LLMError) as info:
        asyncio.run(run_stream(brain))
    assert info.value.reason == "no_provider"


def test_complete_collects_stream():
    brain = make_brain([ollama()], lambda request: httpx.Response(200, text=OLLAMA_BODY))
    completion = asyncio.run(brain.complete([Message(role="user", content="hi")]))
    assert isinstance(completion, Completion)
    assert completion.text == "Hello there"


def test_system_prompt_renders_facts_memories_and_rules():
    prompt = build_system_prompt(
        name="Jarvis",
        tone="dry",
        facts="name=User, ui_theme=dark (stale: confirm if relevant)",
        memories=["User asked about the weather", "User likes tea"],
        tools_enabled=["web_search", "notes"],
    )
    assert "You are Jarvis" in prompt
    assert "Tone: dry" in prompt
    assert "ui_theme=dark (stale" in prompt
    assert "- User likes tea" in prompt
    assert "Tools available: notes, web_search" in prompt
    assert "{{" not in prompt


def test_system_prompt_uses_custom_template_and_keeps_braces():
    prompt = build_system_prompt(
        name="Friday",
        facts="x=1",
        template="Assistant {{name}}.\nFacts:\n{{facts}}\nJSON example: {\"a\": 1}",
    )
    assert prompt == 'Assistant Friday.\nFacts:\nx=1\nJSON example: {"a": 1}'


def test_load_system_template_reads_markdown(tmp_path):
    assert load_system_template(tmp_path) is None
    (tmp_path / "system.md").write_text("  Custom {{name}}  ", encoding="utf-8")
    assert load_system_template(tmp_path) == "Custom {{name}}"
