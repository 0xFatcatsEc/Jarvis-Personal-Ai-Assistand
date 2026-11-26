"""Tests for memory.py: seeding, decay, recall, extraction and the facade."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from config import ChromaConfig, MemoryConfig
from memory import (
    CORE_KEYS,
    ChromaConversations,
    FactStore,
    Memory,
    NullConversations,
    RecallContext,
    Turn,
    format_facts,
    parse_facts,
)

SEEDS = {
    "name": "User",
    "timezone": "America/New_York",
    "preferred_language": "English",
    "preferred_tts_voice": "en_US-lessac-medium",
    "home_assistant_url": "http://localhost:8123",
    "assistant_name": "Jarvis",
    "tone": "concise, slightly dry humor, like a competent butler",
}


class Clock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, days: float) -> None:
        self.now += timedelta(days=days)


@pytest.fixture()
def clock() -> Clock:
    return Clock()


@pytest.fixture()
def store(tmp_path, clock) -> FactStore:
    return FactStore(tmp_path / "facts.db", seeds=SEEDS, clock=clock)


def test_seeds_are_written_once(store, tmp_path, clock):
    assert len(store.all()) == 7
    assert store.get("name").value == "User"
    again = FactStore(tmp_path / "facts.db", seeds=SEEDS, clock=clock)
    assert again.seed() == 0
    assert len(again.all()) == 7


def test_seeding_adds_only_missing_keys(store, tmp_path, clock):
    store.upsert("name", "Ada")
    later = FactStore(tmp_path / "facts.db", seeds={**SEEDS, "city": "Boston"}, clock=clock)
    assert later.seed() == 1
    assert later.get("name").value == "Ada"
    assert later.get("city").value == "Boston"


def test_upsert_keeps_created_at_and_clears_stale(store, clock):
    fact = store.upsert("ui_theme", "dark")
    created = fact.created_at
    clock.advance(100)
    store.decay()
    assert store.get("ui_theme").stale is True
    fact = store.upsert("ui_theme", "light")
    assert fact.created_at == created
    assert fact.stale is False
    assert fact.value == "light"


@pytest.mark.parametrize("key", ["Bad Key", "UPPER", "9lives", "", "x" * 60])
def test_upsert_rejects_bad_keys(store, key):
    with pytest.raises(ValueError):
        store.upsert(key, "x")


def test_upsert_rejects_empty_value(store):
    with pytest.raises(ValueError):
        store.upsert("ui_theme", "   ")


def test_decay_flags_only_old_facts(store, clock):
    store.seed()  # seeds land at t0 and are connected before time moves
    clock.advance(10)
    store.upsert("ui_theme", "dark")  # fresh at t0+10
    clock.advance(85)  # t0+95: cutoff is t0+5, so seeds are old, ui_theme is not
    assert store.decay() == len(SEEDS)
    assert store.get("ui_theme").stale is False
    assert store.decay() == 0


def test_touch_clears_stale(store, clock):
    store.upsert("ui_theme", "dark")
    clock.advance(100)
    store.decay()
    assert store.get("ui_theme").stale
    store.touch(["ui_theme"])
    fact = store.get("ui_theme")
    assert fact.stale is False and fact.last_accessed is not None


def test_recall_includes_core_facts_then_topic_matches(store):
    store.upsert("favourite_tea", "earl grey")
    store.upsert("car_model", "volvo")
    facts = store.recall("what tea do I like?", top_k=8)
    keys = [f.key for f in facts]
    assert keys[: len(CORE_KEYS)] == list(CORE_KEYS)
    assert "favourite_tea" in keys
    assert "car_model" not in keys


def test_recall_touches_returned_facts(store):
    store.upsert("favourite_tea", "earl grey")
    store.recall("tea")
    assert store.get("favourite_tea").last_accessed is not None


def test_recall_respects_top_k(store):
    for index in range(20):
        store.upsert(f"habit_{index}", f"reading book {index}")
    facts = store.recall("reading book", top_k=5)
    assert len(facts) == 5


def test_format_facts_marks_stale(store, clock):
    store.seed()
    clock.advance(200)
    store.decay()
    text = format_facts([store.get("name"), store.get("timezone")])
    assert text == "name=User (stale: confirm if relevant), timezone=America/New_York (stale: confirm if relevant)"


def test_parse_facts_accepts_array_and_code_fence():
    raw = '```json\n[{"key": "ui_theme", "value": "dark"}, {"key": "City", "value": "Boston"}]\n```'
    assert parse_facts(raw) == [("ui_theme", "dark"), ("city", "Boston")]


def test_parse_facts_accepts_wrapped_object():
    assert parse_facts('{"facts": [{"key": "ui_theme", "value": "dark"}]}') == [("ui_theme", "dark")]


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", "{}", '[{"key": "bad key!", "value": "x"}]'])
def test_parse_facts_rejects_junk(raw):
    assert parse_facts(raw) == []


def test_parse_facts_skips_sensitive_and_known_values():
    raw = (
        '[{"key": "password", "value": "hunter2"},'
        ' {"key": "api_key", "value": "sk-123"},'
        ' {"key": "ui_theme", "value": "dark"}]'
    )
    assert parse_facts(raw, known={"ui_theme": "dark"}) == []


def test_parse_facts_caps_and_dedupes():
    items = ",".join(f'{{"key": "k{i}", "value": "v{i}"}}' for i in range(10))
    items += ',{"key": "k9", "value": "override"}'
    parsed = parse_facts(f"[{items}]", max_new=3)
    assert len(parsed) == 3
    parsed = parse_facts('[{"key":"k","value":"one"},{"key":"k","value":"two"}]')
    assert parsed == [("k", "two")]


class FakeConversations:
    def __init__(self):
        self.added: list[Turn] = []
        self.searches: list[tuple] = []
        self.fail = False

    def add_turn(self, turn: Turn) -> None:
        self.added.append(turn)

    def search(self, query, *, top_k, min_similarity):
        self.searches.append((query, top_k, min_similarity))
        if self.fail:
            raise RuntimeError("chroma down")
        return [Turn(user="earlier", assistant="reply", ts="2026-01-01T00:00:00+00:00", similarity=0.8)]

    def count(self):
        return len(self.added)

    def close(self):
        pass


def memory_config(tmp_path) -> MemoryConfig:
    return MemoryConfig(
        sqlite_path=tmp_path / "mem.db",
        seed_facts=SEEDS,
        chroma=ChromaConfig(path=tmp_path / "chroma"),
    )


def test_memory_recall_context_uses_config_limits(tmp_path):
    conversations = FakeConversations()
    memory = Memory(memory_config(tmp_path), conversations=conversations)
    context = memory.recall_context("what tea?")
    assert isinstance(context, RecallContext)
    assert [f.key for f in context.facts][: len(CORE_KEYS)] == list(CORE_KEYS)
    assert conversations.searches == [("what tea?", 5, 0.25)]
    assert context.memory_lines() == ["User: earlier -> Assistant: reply"]


def test_memory_survives_recall_failure(tmp_path):
    conversations = FakeConversations()
    conversations.fail = True
    memory = Memory(memory_config(tmp_path), conversations=conversations)
    context = memory.recall_context("anything")
    assert context.turns == []


def test_memory_remember_turn(tmp_path):
    conversations = FakeConversations()
    memory = Memory(memory_config(tmp_path), conversations=conversations)
    memory.remember_turn("hello", "hi")
    assert conversations.added[0].user == "hello"
    assert conversations.added[0].ts


def test_learn_stores_new_facts(tmp_path):
    memory = Memory(memory_config(tmp_path), conversations=FakeConversations())

    async def fake_llm(prompt: str) -> str:
        assert "Known facts:" in prompt and "name=User" in prompt
        return '[{"key": "favourite_tea", "value": "earl grey"}]'

    stored = asyncio.run(memory.learn("I love earl grey", "Noted.", fake_llm))
    assert [f.key for f in stored] == ["favourite_tea"]
    assert memory.get("favourite_tea").value == "earl grey"


def test_learn_skips_unchanged_and_disabled(tmp_path):
    cfg = memory_config(tmp_path)
    cfg.learn.max_new_facts = 5
    memory = Memory(cfg, conversations=FakeConversations())

    async def unchanged(prompt: str) -> str:
        return '[{"key": "name", "value": "User"}]'

    assert asyncio.run(memory.learn("I am User", "ok", unchanged)) == []

    cfg.learn.enabled = False
    disabled = Memory(cfg, conversations=FakeConversations())
    assert asyncio.run(disabled.learn("I like tea", "ok", unchanged)) == []


def test_learn_survives_llm_failure(tmp_path):
    memory = Memory(memory_config(tmp_path), conversations=FakeConversations())

    async def broken(prompt: str) -> str:
        raise RuntimeError("model down")

    assert asyncio.run(memory.learn("I like tea", "ok", broken)) == []


def test_null_conversations_is_inert():
    store = NullConversations()
    store.add_turn(Turn(user="a", assistant="b"))
    assert store.search("a", top_k=3, min_similarity=0.2) == []
    assert store.count() == 0


def test_chroma_document_round_trip():
    turn = Turn(user="what is the weather", assistant="sunny", ts="2026-01-01T00:00:00+00:00")
    document = turn.text()
    assert document.startswith("User: what is the weather")
    assert "Assistant: sunny" in document


def test_delete_removes_existing_fact(store):
    assert store.delete("tone") is True
    assert store.get("tone") is None
    assert "tone" not in [fact.key for fact in store.all()]


def test_delete_missing_fact_returns_false(store):
    assert store.delete("no_such_key") is False
