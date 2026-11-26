"""Tests for tools.py: search, notes, reminders, Home Assistant, shell, registry."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from config import (
    HomeAssistantConfig,
    NotesConfig,
    RemindersConfig,
    ShellConfig,
    ToolsConfig,
    WebSearchConfig,
)
from tools import (
    HomeAssistant,
    Notes,
    ReminderStore,
    Reminders,
    Shell,
    ToolError,
    ToolRegistry,
    WebSearch,
    parse_when,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


def tools_config(tmp_path: Path, *, enabled=None, whitelist=("echo", "pwd")) -> ToolsConfig:
    return ToolsConfig(
        enabled=list(enabled or ["web_search", "notes", "reminders", "home_assistant", "shell"]),
        web_search=WebSearchConfig(provider="ddgs"),
        home_assistant=HomeAssistantConfig(url="http://ha.test", token="tok", allow_domains=["light", "switch", "scene"]),
        notes=NotesConfig(dir=tmp_path / "notes"),
        reminders=RemindersConfig(db_path=tmp_path / "reminders.db"),
        shell=ShellConfig(whitelist=list(whitelist)),
    )


# --------------------------------------------------------------------------- #
# web search
# --------------------------------------------------------------------------- #
def test_ddgs_search_formats_results():
    def search(query, limit):
        assert query == "weather" and limit == 2
        return [
            {"title": "Forecast", "href": "http://a.test", "body": " Sunny   all day "},
            {"title": "Radar", "href": "http://b.test", "body": "Rain later"},
        ]

    tool = WebSearch(WebSearchConfig(provider="ddgs"), ddgs_search=search)
    text = run(tool.run({"query": "weather", "max_results": 2}))
    assert "1. Forecast — http://a.test" in text
    assert "Sunny all day" in text
    assert "2. Radar — http://b.test" in text


def test_ddgs_no_results():
    tool = WebSearch(WebSearchConfig(provider="ddgs"), ddgs_search=lambda q, n: [])
    assert run(tool.run({"query": "nothing"})) == "No results."


def test_tavily_posts_key_and_query():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        assert request.url == httpx.URL("https://api.tavily.com/search")
        return httpx.Response(200, json={"results": [{"title": "T", "url": "http://t.test", "content": "C"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = WebSearch(WebSearchConfig(provider="tavily", api_key="tv-key"), client=client)
    text = run(tool.run({"query": "news"}))
    assert "1. T — http://t.test" in text
    assert seen["body"]["api_key"] == "tv-key"
    assert seen["body"]["query"] == "news"
    run(client.aclose())


def test_tavily_without_key_is_explicit():
    tool = WebSearch(WebSearchConfig(provider="tavily"))
    with pytest.raises(ToolError, match="TAVILY_API_KEY"):
        run(tool.run({"query": "x"}))


def test_searxng_requires_url():
    with pytest.raises(Exception, match="searxng_url"):
        WebSearchConfig(provider="searxng")


def test_searxng_parses_results():
    def handler(request):
        assert request.url.params["q"] == "python"
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json={"results": [{"title": "Py", "url": "http://py.test", "content": "Docs"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = WebSearch(WebSearchConfig(provider="searxng", searxng_url="http://searx.test"), client=client)
    assert "Py — http://py.test" in run(tool.run({"query": "python"}))
    run(client.aclose())


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
def test_notes_round_trip(tmp_path):
    tool = Notes(NotesConfig(dir=tmp_path / "notes"))
    assert "Saved" in run(tool.run({"action": "write", "filename": "Shopping List", "content": "- milk"}))
    assert run(tool.run({"action": "read", "filename": "Shopping List"})).strip() == "- milk"
    assert "Shopping List" in run(tool.run({"action": "list"}))
    run(tool.run({"action": "append", "filename": "Shopping List", "content": "- eggs"}))
    content = (tmp_path / "notes" / "Shopping List.md").read_text()
    assert "- milk" in content and "- eggs" in content


def test_notes_extension_is_not_doubled(tmp_path):
    tool = Notes(NotesConfig(dir=tmp_path / "notes"))
    run(tool.run({"action": "write", "filename": "ideas.md", "content": "x"}))
    assert (tmp_path / "notes" / "ideas.md").exists()


@pytest.mark.parametrize("name", ["../evil", "a/b", "", "x" * 80, ".hidden"])
def test_notes_reject_bad_names(tmp_path, name):
    tool = Notes(NotesConfig(dir=tmp_path / "notes"))
    with pytest.raises(ToolError):
        run(tool.run({"action": "read", "filename": name}))


def test_notes_read_missing(tmp_path):
    tool = Notes(NotesConfig(dir=tmp_path / "notes"))
    with pytest.raises(ToolError, match="does not exist"):
        run(tool.run({"action": "read", "filename": "ghost"}))


def test_notes_enforce_size_limit(tmp_path):
    tool = Notes(NotesConfig(dir=tmp_path / "notes", max_bytes=1000))
    with pytest.raises(ToolError, match="exceed"):
        run(tool.run({"action": "write", "filename": "big", "content": "x" * 1200}))


# --------------------------------------------------------------------------- #
# reminders
# --------------------------------------------------------------------------- #
def test_parse_when_relative_and_clock():
    assert parse_when("in 10 minutes", now=NOW) == datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)
    assert parse_when("2h30m", now=NOW) == datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    assert parse_when("+90s", now=NOW) == datetime(2026, 1, 1, 12, 1, 30, tzinfo=timezone.utc)
    assert parse_when("7pm", now=NOW) == datetime(2026, 1, 1, 19, 0, tzinfo=timezone.utc)
    assert parse_when("12am", now=NOW) == datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert parse_when("12pm", now=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)) == datetime(
        2026, 1, 1, 12, 0, tzinfo=timezone.utc
    )
    assert parse_when("at 9:15 am", now=NOW) == datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
    assert parse_when("tomorrow 9am", now=NOW) == datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc)
    assert parse_when("2026-03-01T10:00:00Z", now=NOW) == datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("raw", ["", "sometime", "0m", "99pm"])
def test_parse_when_rejects_junk(raw):
    with pytest.raises(ValueError):
        parse_when(raw, now=NOW)


def test_reminder_store_due_is_fire_once(tmp_path):
    store = ReminderStore(RemindersConfig(db_path=tmp_path / "r.db"), clock=lambda: NOW)
    reminder = store.add("stretch", parse_when("in 10 minutes", now=NOW))
    assert [r.id for r in store.pending()] == [reminder.id]
    assert store.due(NOW) == []
    fired = store.due(datetime(2026, 1, 1, 12, 11, tzinfo=timezone.utc))
    assert [r.text for r in fired] == ["stretch"]
    assert store.due(datetime(2026, 1, 1, 12, 12, tzinfo=timezone.utc)) == []
    assert store.pending() == []


def test_reminder_store_caps_pending(tmp_path):
    store = ReminderStore(RemindersConfig(db_path=tmp_path / "r.db", max_per_user=2), clock=lambda: NOW)
    store.add("one", NOW)
    store.add("two", NOW)
    with pytest.raises(ToolError, match="too many"):
        store.add("three", NOW)


def test_reminder_tool_add_list_cancel(tmp_path):
    store = ReminderStore(RemindersConfig(db_path=tmp_path / "r.db"), clock=lambda: NOW)
    tool = Reminders(store)
    added = run(tool.run({"action": "add", "text": "call mom", "when": "in 5 minutes"}))
    assert "call mom" in added
    assert "call mom" in run(tool.run({"action": "list"}))
    with pytest.raises(ToolError, match="numeric"):
        run(tool.run({"action": "cancel", "id": "abc"}))
    with pytest.raises(ToolError, match="no pending"):
        run(tool.run({"action": "cancel", "id": 999}))


# --------------------------------------------------------------------------- #
# Home Assistant
# --------------------------------------------------------------------------- #
def ha_client(seen: dict) -> httpx.AsyncClient:
    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        if request.url.path == "/api/states":
            return httpx.Response(
                200,
                json=[
                    {"entity_id": "light.kitchen", "state": "on", "attributes": {"friendly_name": "Kitchen"}},
                    {"entity_id": "switch.fan", "state": "off", "attributes": {}},
                    {"entity_id": "sensor.temp", "state": "21", "attributes": {}},
                ],
            )
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_home_assistant_lists_allowed_domains():
    seen = {}
    client = ha_client(seen)
    tool = HomeAssistant(
        HomeAssistantConfig(url="http://ha.test", token="tok", allow_domains=["light", "switch"]), client=client
    )
    text = run(tool.run({"action": "list_states"}))
    assert "light.kitchen: on (Kitchen)" in text
    assert "switch.fan: off" in text
    assert "sensor.temp" not in text
    assert seen["auth"] == "Bearer tok"
    run(client.aclose())


def test_home_assistant_turn_on_posts_service():
    seen = {}
    client = ha_client(seen)
    tool = HomeAssistant(HomeAssistantConfig(url="http://ha.test/", token="tok"), client=client)
    assert run(tool.run({"action": "turn_on", "entity_id": "light.kitchen"})) == "turn_on sent to light.kitchen."
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/services/light/turn_on"
    assert seen["body"] == {"entity_id": "light.kitchen"}
    run(client.aclose())


def test_home_assistant_toggle_scene_uses_turn_on():
    seen = {}
    client = ha_client(seen)
    tool = HomeAssistant(HomeAssistantConfig(url="http://ha.test", token="tok", allow_domains=["scene"]), client=client)
    run(tool.run({"action": "toggle", "entity_id": "scene.movie"}))
    assert seen["path"] == "/api/services/scene/turn_on"
    run(client.aclose())


@pytest.mark.parametrize(
    "args, message",
    [
        ({"action": "turn_on", "entity_id": "camera.front"}, "not allowed"),
        ({"action": "turn_on", "entity_id": "kitchen"}, "must look like"),
        ({"action": "explode"}, "action must be"),
    ],
)
def test_home_assistant_rejects_bad_calls(args, message):
    tool = HomeAssistant(HomeAssistantConfig(url="http://ha.test", token="tok"))
    with pytest.raises(ToolError, match=message):
        run(tool.run(args))


def test_home_assistant_requires_configuration():
    with pytest.raises(ToolError, match="not configured"):
        run(HomeAssistant(HomeAssistantConfig(url=None, token="tok")).run({"action": "list_states"}))
    with pytest.raises(ToolError, match="not configured"):
        run(HomeAssistant(HomeAssistantConfig(url="http://ha.test", token=None)).run({"action": "list_states"}))


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
def test_shell_runs_whitelisted_command():
    tool = Shell(ShellConfig(whitelist=["echo"], timeout_s=5))
    output = run(tool.run({"command": "echo hello"}))
    assert "hello" in output and "exit 0" in output


def test_shell_rejects_non_whitelisted_command():
    tool = Shell(ShellConfig(whitelist=["echo"]))
    with pytest.raises(ToolError, match="not whitelisted"):
        run(tool.run({"command": "rm -rf data"}))


@pytest.mark.parametrize("command", ["ls /etc", "ls ../.."])
def test_shell_rejects_absolute_and_parent_paths(command):
    tool = Shell(ShellConfig(whitelist=["ls"]))
    with pytest.raises(ToolError, match="not allowed"):
        run(tool.run({"command": command}))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_only_exposes_enabled_tools(tmp_path):
    registry = ToolRegistry(tools_config(tmp_path, enabled=["notes", "shell"]))
    assert registry.names() == ["notes", "shell"]
    schemas = registry.schemas()
    assert {schema["name"] for schema in schemas} == {"notes", "shell"}
    assert all(schema["parameters"]["type"] == "object" for schema in schemas)


def test_registry_unknown_and_disabled_tools_error(tmp_path):
    registry = ToolRegistry(tools_config(tmp_path, enabled=["notes"]))
    with pytest.raises(ToolError, match="not available"):
        run(registry.call("shell", {"command": "ls"}))
    with pytest.raises(ToolError, match="not available"):
        run(registry.call("nope", {}))


def test_registry_marks_side_effects_risky(tmp_path):
    registry = ToolRegistry(tools_config(tmp_path))
    assert registry.risky("notes", {"action": "read"}) is False
    assert registry.risky("notes", {"action": "write"}) is True
    assert registry.risky("home_assistant", {"action": "list_states"}) is False
    assert registry.risky("home_assistant", {"action": "turn_on"}) is True
    assert registry.risky("web_search", {"query": "x"}) is False


def test_registry_describe_mentions_target(tmp_path):
    registry = ToolRegistry(tools_config(tmp_path))
    assert registry.describe("shell", {"command": "df -h"}) == "run 'df -h'"
    assert "light.kitchen" in registry.describe("home_assistant", {"action": "turn_on", "entity_id": "light.kitchen"})


def test_registry_wraps_unexpected_errors(tmp_path):
    registry = ToolRegistry(tools_config(tmp_path))

    async def boom(args):
        raise RuntimeError("kaboom")

    registry._tools["web_search"].handler = boom
    with pytest.raises(ToolError, match="kaboom"):
        run(registry.call("web_search", {"query": "x"}))
