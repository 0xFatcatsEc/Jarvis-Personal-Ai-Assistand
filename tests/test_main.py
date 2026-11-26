"""Tests for main.py: conversation flow, tools, confirmation, prompts, CLI."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from brain import AllProvidersFailed, Completion, Message, TextDelta, ToolCall
from config import Config
from main import MAX_TOOL_ROUNDS, Jarvis, _matches, check_summary, parse_args
from memory import Fact, RecallContext


def run(coro):
    return asyncio.run(coro)


class FakeBrain:
    def __init__(self, completions=None, fail=False):
        self.queue = list(completions or [])
        self.fail = fail
        self.stream_calls: list[list[Message]] = []
        self.extract_calls = 0
        self.closed = False

    async def stream(self, messages, tools=None, max_tokens=None, temperature=None):
        self.stream_calls.append(list(messages))
        if self.fail:
            raise AllProvidersFailed([])
            yield  # pragma: no cover
        completion = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if completion.text:
            yield TextDelta(completion.text)
        yield completion

    async def complete(self, messages, tools=None, max_tokens=None, temperature=None):
        self.extract_calls += 1
        return Completion(text="[]", provider="fake", model="fake")

    async def aclose(self):
        self.closed = True


class FakeMemory:
    def __init__(self, facts=None):
        self.facts = facts or []
        self.queries: list[str] = []
        self.turns: list[tuple[str, str]] = []
        self.learned: list[tuple[str, str]] = []
        self.closed = False

    def recall_context(self, query: str) -> RecallContext:
        self.queries.append(query)
        return RecallContext(facts=self.facts, turns=[])

    def remember_turn(self, user, assistant, ts=None):
        self.turns.append((user, assistant))

    async def learn(self, user, assistant, llm):
        self.learned.append((user, assistant))
        self.extracted_raw = await llm("extract facts")
        return []

    def close(self):
        self.closed = True


class FakeTools:
    def __init__(self, risky_names=("notes",)):
        self.risky_names = set(risky_names)
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def names(self):
        return ["notes", "web_search"]

    def schemas(self):
        return [
            {"name": "notes", "description": "notes", "parameters": {"type": "object", "properties": {}}},
            {"name": "web_search", "description": "search", "parameters": {"type": "object", "properties": {}}},
        ]

    def risky(self, name, args):
        if name == "notes":
            return str(args.get("action", "")).lower() in {"write", "append"}
        return name in self.risky_names and name != "notes"

    def describe(self, name, args):
        if name == "notes":
            return f"{args.get('action')} note {args.get('filename')!r}"
        return f"{name} {args.get('query', '')}".strip()

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return f"{name} done"

    def close(self):
        self.closed = True


class FakeSpeaker:
    def __init__(self, available=True):
        self.available = available
        self.spoken: list[str] = []

    def speak_chunks(self, text):
        self.spoken.append(text)


def make_jarvis(completions, *, facts=None, fail=False, speaker=None, write=None):
    cfg = Config()
    brain = FakeBrain(completions, fail=fail)
    memory = FakeMemory(facts)
    tools = FakeTools()
    jarvis = Jarvis(
        cfg,
        brain=brain,
        memory=memory,
        tools=tools,
        speaker=speaker,
        write=write or (lambda text: None),
    )
    return jarvis, brain, memory, tools


def test_reply_is_stored_and_learned_from():
    jarvis, brain, memory, _ = make_jarvis([Completion("Hello, sir.", "fake", "fake")])
    reply = run(jarvis.handle("hi"))
    assert reply == "Hello, sir."
    assert memory.turns == [("hi", "Hello, sir.")]
    assert memory.learned == [("hi", "Hello, sir.")]
    assert memory.queries == ["hi"]
    assert brain.extract_calls == 1
    assert [m.role for m in jarvis._history] == ["user", "assistant"]


def test_speaks_reply_when_asked():
    speaker = FakeSpeaker()
    jarvis, *_ = make_jarvis([Completion("Morning.", "fake", "fake")], speaker=speaker)
    run(jarvis.handle("hello", speak=True))
    assert speaker.spoken == ["Morning."]


def test_facts_and_tone_reach_the_system_prompt():
    facts = [
        Fact("name", "User", "2026-01-01T00:00:00+00:00"),
        Fact("tone", "dry wit", "2026-01-01T00:00:00+00:00"),
        Fact("assistant_name", "Jeeves", "2026-01-01T00:00:00+00:00"),
    ]
    jarvis, brain, *_ = make_jarvis([Completion("Very good.", "fake", "fake")], facts=facts)
    run(jarvis.handle("status"))
    system = brain.stream_calls[0][0]
    assert system.role == "system"
    assert "You are Jeeves" in system.content
    assert "Tone: dry wit" in system.content
    assert "name=User" in system.content


def test_tool_call_round_trip_feeds_result_back_to_model():
    completions = [
        Completion("", "fake", "fake", [ToolCall("c1", "web_search", {"query": "weather"})]),
        Completion("Sunny all day.", "fake", "fake"),
    ]
    jarvis, brain, memory, tools = make_jarvis(completions)
    reply = run(jarvis.handle("weather?"))
    assert reply == "Sunny all day."
    assert tools.calls == [("web_search", {"query": "weather"})]
    second_call = brain.stream_calls[1]
    tool_message = second_call[-1]
    assert tool_message.role == "tool"
    assert tool_message.content == "web_search done"
    assert tool_message.tool_call_id == "c1"
    assert memory.turns == [("weather?", "Sunny all day.")]


def test_risky_tool_asks_then_runs_on_yes():
    completions = [
        Completion("", "fake", "fake", [ToolCall("c1", "notes", {"action": "write", "filename": "ideas", "content": "x"})]),
        Completion("Saved.", "fake", "fake"),
    ]
    speaker = FakeSpeaker()
    jarvis, brain, memory, tools = make_jarvis(completions, speaker=speaker)

    prompt = run(jarvis.handle("save a note", speak=True))
    assert prompt == "Should I write note 'ideas'? Say yes to confirm."
    assert tools.calls == []
    assert speaker.spoken == [prompt]
    assert jarvis._pending is not None

    reply = run(jarvis.handle("yes"))
    assert reply == "Saved."
    assert tools.calls == [("notes", {"action": "write", "filename": "ideas", "content": "x"})]
    assert memory.turns == [("save a note", "Saved.")]
    assert jarvis._pending is None


def test_no_cancels_pending_tool():
    completions = [Completion("", "fake", "fake", [ToolCall("c1", "notes", {"action": "write", "filename": "x"})])]
    jarvis, _, memory, tools = make_jarvis(completions)
    run(jarvis.handle("write it"))
    assert run(jarvis.handle("no, cancel")) == "Okay, cancelled."
    assert tools.calls == []
    assert jarvis._pending is None
    assert memory.turns == []


def test_unrelated_message_cancels_pending_and_answers():
    completions = [
        Completion("", "fake", "fake", [ToolCall("c1", "notes", {"action": "write", "filename": "x"})]),
        Completion("It is noon.", "fake", "fake"),
    ]
    jarvis, _, _, tools = make_jarvis(completions)
    run(jarvis.handle("write it"))
    assert run(jarvis.handle("what time is it?")) == "It is noon."
    assert tools.calls == []
    assert jarvis._pending is None


def test_llm_failure_is_handled_gently():
    speaker = FakeSpeaker()
    jarvis, *_ = make_jarvis([], fail=True, speaker=speaker)
    reply = run(jarvis.handle("hello", speak=True))
    assert reply == "I can't reach a language model right now."
    assert speaker.spoken == [reply]


def test_tool_round_limit_stops_the_loop():
    tool_call = ToolCall("c", "web_search", {"query": "loop"})
    completions = [Completion("", "fake", "fake", [tool_call])]
    jarvis, brain, _, tools = make_jarvis(completions)
    reply = run(jarvis.handle("go"))
    assert reply == "I stopped after several tool calls without reaching an answer."
    assert len(tools.calls) == MAX_TOOL_ROUNDS
    assert len(brain.stream_calls) == MAX_TOOL_ROUNDS


def test_once_prints_reply():
    printed = []
    jarvis, *_ = make_jarvis([Completion("Hi.", "fake", "fake")], write=printed.append)
    assert run(jarvis.once("hello")) == "Hi."
    assert "Jarvis: Hi." in printed


@pytest.mark.parametrize("text", ["yes", "Yes", "yeah", "sure", "go ahead"])
def test_affirmatives(text):
    assert _matches(text, r"^(yes|yeah|yep|yup|sure|ok|okay|do it|go ahead|please do|confirm)\b")


@pytest.mark.parametrize("text", ["no", "Nope", "cancel", "never mind"])
def test_negatives(text):
    assert _matches(text, r"^(no|nope|nah|cancel|never mind|nevermind|don'?t|stop)\b")


def test_config_check_summary():
    summary = check_summary(Config())
    assert "config OK: Jarvis" in summary
    assert "tools" in summary


def test_cli_flags():
    args = parse_args(["--text", "--config", "custom.yaml"])
    assert args.text is True and args.config == "custom.yaml"
    assert args.once is None
    assert parse_args(["--once", "hello"]).once == "hello"


def test_setup_logging_rotates_logs(tmp_path):
    import logging
    from logging.handlers import RotatingFileHandler

    from config import AssistantConfig
    from main import setup_logging

    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        setup_logging(Config(assistant=AssistantConfig(log_dir=tmp_path)))
        files = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert len(files) == 1
        assert files[0].maxBytes == 2_000_000
        assert files[0].backupCount == 3
        assert (tmp_path / "jarvis.log").exists()
    finally:
        root.handlers.clear()
        root.handlers.extend(saved)
