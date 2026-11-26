"""Tools Jarvis can call: web search, notes, reminders, Home Assistant, shell.

Each tool is a JSON-schema description plus an async handler returning text for
the LLM. `ToolRegistry` only builds the tools listed in `tools.enabled`, and
marks the ones with side effects so `main` can ask for confirmation first.

Everything external (HTTP, ddgs, subprocess, clock) is injectable: the registry
works the same in tests and on the user's machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from config import HomeAssistantConfig, NotesConfig, RemindersConfig, ShellConfig, ToolsConfig, WebSearchConfig

log = logging.getLogger(__name__)

NOTE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
RELATIVE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)(?![a-z])"
)
CLOCK_RE = re.compile(r"^(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")
UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class ToolError(RuntimeError):
    """A tool refused or failed; the message is safe to show the model."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… (truncated at {limit} chars)"


def _str_arg(args: dict[str, Any], key: str, *, required: bool = True, max_len: int = 500) -> str:
    value = args.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ToolError(f"missing required argument {key!r}")
        return ""
    if not isinstance(value, str):
        raise ToolError(f"argument {key!r} must be text")
    return value.strip()[:max_len]


def _int_arg(args: dict[str, Any], key: str, default: int, *, low: int, high: int) -> int:
    value = args.get(key, default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"argument {key!r} must be an integer") from None
    return max(low, min(high, number))


# --------------------------------------------------------------------------- #
# web search
# --------------------------------------------------------------------------- #
class WebSearch:
    def __init__(
        self,
        cfg: WebSearchConfig,
        *,
        client: httpx.AsyncClient | None = None,
        ddgs_search: Callable[[str, int], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.cfg = cfg
        self._client = client
        self._ddgs = ddgs_search

    async def run(self, args: dict[str, Any]) -> str:
        query = _str_arg(args, "query", max_len=300)
        limit = _int_arg(args, "max_results", self.cfg.max_results, low=1, high=10)
        if self.cfg.provider == "ddgs":
            results = await asyncio.to_thread(self._search_ddgs, query, limit)
        elif self.cfg.provider == "tavily":
            results = await self._search_tavily(query, limit)
        else:
            results = await self._search_searxng(query, limit)
        if not results:
            return "No results."
        lines = []
        for index, item in enumerate(results[:limit], start=1):
            title = str(item.get("title", "")).strip() or "(untitled)"
            url = str(item.get("url") or item.get("href") or "").strip()
            snippet = " ".join(str(item.get("content") or item.get("body") or "").split())
            lines.append(f"{index}. {title} — {url}\n   {_cap(snippet, 400)}")
        return _cap("\n".join(lines), 4000)

    def _search_ddgs(self, query: str, limit: int) -> list[dict[str, Any]]:
        if self._ddgs is not None:
            return self._ddgs(query, limit)
        try:
            from ddgs import DDGS  # current package name
        except ImportError:
            from duckduckgo_search import DDGS  # older name
        with DDGS() as search:
            return list(search.text(query, max_results=limit))

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def _search_tavily(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.cfg.api_key:
            raise ToolError("web search is set to tavily but TAVILY_API_KEY is not set")
        response = await self.client().post(
            "https://api.tavily.com/search",
            json={"api_key": self.cfg.api_key, "query": query, "max_results": limit, "search_depth": "basic"},
            timeout=self.cfg.timeout_s,
        )
        if response.status_code >= 400:
            raise ToolError(f"tavily returned HTTP {response.status_code}")
        payload = response.json()
        return [
            {"title": item.get("title"), "url": item.get("url"), "content": item.get("content")}
            for item in payload.get("results", [])
        ]

    async def _search_searxng(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.cfg.searxng_url:
            raise ToolError("web search is set to searxng but searxng_url is empty")
        response = await self.client().get(
            self.cfg.searxng_url.rstrip("/") + "/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=self.cfg.timeout_s,
        )
        if response.status_code >= 400:
            raise ToolError(f"searxng returned HTTP {response.status_code}")
        payload = response.json()
        return [
            {"title": item.get("title"), "url": item.get("url"), "content": item.get("content")}
            for item in payload.get("results", [])
        ]


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
class Notes:
    def __init__(self, cfg: NotesConfig) -> None:
        self.cfg = cfg
        self.dir = Path(cfg.dir)

    def _path(self, filename: str) -> Path:
        raw = filename.strip()
        if raw.lower().endswith(".md"):
            raw = raw[:-3]
        if not NOTE_NAME_RE.fullmatch(raw) or ".." in raw:
            raise ToolError(f"invalid note name {filename!r}; use letters, digits, spaces, dots, dashes")
        path = (self.dir / f"{raw}.md").resolve()
        if path.parent != self.dir.resolve():
            raise ToolError("note path escapes the notes directory")
        return path

    async def run(self, args: dict[str, Any]) -> str:
        action = _str_arg(args, "action", max_len=20).lower()
        if action == "list":
            return self._list()
        filename = _str_arg(args, "filename", max_len=80)
        path = self._path(filename)
        if action == "read":
            return self._read(path)
        if action == "write":
            return self._write(path, _str_arg(args, "content", max_len=self.cfg.max_bytes + 1000))
        if action == "append":
            return self._append(path, _str_arg(args, "content", max_len=self.cfg.max_bytes + 1000))
        raise ToolError("notes action must be list, read, write or append")

    def _list(self) -> str:
        if not self.dir.exists():
            return "No notes yet."
        notes = sorted(self.dir.glob("*.md"))
        if not notes:
            return "No notes yet."
        return "\n".join(f"- {note.stem} ({note.stat().st_size} bytes)" for note in notes[:100])

    def _read(self, path: Path) -> str:
        if not path.exists():
            raise ToolError(f"note {path.stem!r} does not exist")
        if path.stat().st_size > self.cfg.max_bytes:
            raise ToolError(f"note {path.stem!r} is larger than {self.cfg.max_bytes} bytes")
        return _cap(path.read_text(encoding="utf-8", errors="replace"), self.cfg.max_bytes)

    def _write(self, path: Path, content: str) -> str:
        if len(content.encode("utf-8")) > self.cfg.max_bytes:
            raise ToolError(f"note would exceed {self.cfg.max_bytes} bytes")
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return f"Saved note {path.stem!r} ({path.stat().st_size} bytes)."

    def _append(self, path: Path, content: str) -> str:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        combined = (existing.rstrip() + "\n\n" + content.strip() + "\n") if existing else content.rstrip() + "\n"
        if len(combined.encode("utf-8")) > self.cfg.max_bytes:
            raise ToolError(f"note would exceed {self.cfg.max_bytes} bytes")
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(combined, encoding="utf-8")
        return f"Appended to note {path.stem!r} ({path.stat().st_size} bytes)."


# --------------------------------------------------------------------------- #
# reminders
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Reminder:
    id: int
    text: str
    due_at: str
    created_at: str


def parse_when(raw: str, *, now: datetime | None = None) -> datetime:
    """'in 10 minutes', '2h30m', '7pm', 'tomorrow 9am', or ISO -> aware UTC."""
    text = " ".join(raw.strip().lower().split())
    if not text:
        raise ValueError("empty time")
    local_now = now or datetime.now().astimezone()
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=timezone.utc)

    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=local_now.tzinfo)
        return parsed.astimezone(timezone.utc)

    day_offset = 0
    body = text
    if body.startswith("tomorrow"):
        day_offset = 1
        body = body.removeprefix("tomorrow").strip()

    relative = body
    if relative.startswith("in "):
        relative = relative[3:].strip()
    if relative.startswith("+"):
        relative = relative[1:].strip()
    parts = RELATIVE_RE.findall(relative)
    if parts and not RELATIVE_RE.sub("", relative).strip(" .,"):
        seconds = 0.0
        for amount, unit in parts:
            seconds += float(amount) * UNIT_SECONDS[unit[0]]
        if seconds <= 0:
            raise ValueError("relative time must be positive")
        return (local_now + timedelta(seconds=seconds)).astimezone(timezone.utc)

    clock = CLOCK_RE.fullmatch(body)
    if clock:
        hour = int(clock.group(1))
        minute = int(clock.group(2) or 0)
        meridiem = clock.group(3)
        if meridiem:
            if not 1 <= hour <= 12:
                raise ValueError("clock time out of range")
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("clock time out of range")
        target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=day_offset)
        if target <= local_now:
            target += timedelta(days=1)
        return target.astimezone(timezone.utc)

    raise ValueError(f"cannot understand time {raw!r}")


class ReminderStore:
    def __init__(self, cfg: RemindersConfig, *, clock: Callable[[], datetime] | None = None) -> None:
        self.cfg = cfg
        self.path = Path(cfg.db_path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    text       TEXT NOT NULL,
                    due_at     TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    fired      INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS reminders_due ON reminders (fired, due_at);
                """
            )
            db.commit()
            self._db = db
        return self._db

    def add(self, text: str, due_at: datetime) -> Reminder:
        text = text.strip()
        if not text:
            raise ToolError("reminder text is empty")
        now = self._clock().astimezone(timezone.utc)
        due_iso = due_at.astimezone(timezone.utc).replace(microsecond=0).isoformat()
        with self._lock:
            db = self._connection()
            pending = db.execute("SELECT COUNT(*) FROM reminders WHERE fired = 0").fetchone()[0]
            if pending >= self.cfg.max_per_user:
                raise ToolError(f"too many pending reminders (limit {self.cfg.max_per_user})")
            cursor = db.execute(
                "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
                (text[:500], due_iso, now.replace(microsecond=0).isoformat()),
            )
            db.commit()
            row = db.execute("SELECT * FROM reminders WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return self._row(row)

    def pending(self) -> list[Reminder]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT * FROM reminders WHERE fired = 0 ORDER BY due_at"
            ).fetchall()
        return [self._row(row) for row in rows]

    def cancel(self, reminder_id: int) -> bool:
        with self._lock:
            db = self._connection()
            cursor = db.execute("DELETE FROM reminders WHERE id = ? AND fired = 0", (reminder_id,))
            db.commit()
        return cursor.rowcount > 0

    def due(self, now: datetime | None = None) -> list[Reminder]:
        """Fired-once semantics: due reminders are marked before returning."""
        moment = (now or self._clock()).astimezone(timezone.utc).replace(microsecond=0).isoformat()
        with self._lock:
            db = self._connection()
            rows = db.execute(
                "SELECT * FROM reminders WHERE fired = 0 AND due_at <= ? ORDER BY due_at", (moment,)
            ).fetchall()
            if rows:
                ids = [row["id"] for row in rows]
                db.execute(f"UPDATE reminders SET fired = 1 WHERE id IN ({','.join('?' for _ in ids)})", ids)
                db.commit()
        return [self._row(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @staticmethod
    def _row(row: sqlite3.Row) -> Reminder:
        return Reminder(id=row["id"], text=row["text"], due_at=row["due_at"], created_at=row["created_at"])


class Reminders:
    def __init__(self, store: ReminderStore) -> None:
        self.store = store

    async def run(self, args: dict[str, Any]) -> str:
        action = _str_arg(args, "action", max_len=20).lower()
        if action == "add":
            text = _str_arg(args, "text", max_len=500)
            when = _str_arg(args, "when", max_len=100)
            try:
                due_at = parse_when(when)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            reminder = self.store.add(text, due_at)
            return f"Reminder #{reminder.id} set for {reminder.due_at}: {reminder.text}"
        if action == "list":
            pending = self.store.pending()
            if not pending:
                return "No pending reminders."
            return "\n".join(f"#{item.id} {item.due_at} — {item.text}" for item in pending[:50])
        if action == "cancel":
            try:
                reminder_id = int(args.get("id"))
            except (TypeError, ValueError):
                raise ToolError("cancel needs a numeric reminder id") from None
            if not self.store.cancel(reminder_id):
                raise ToolError(f"no pending reminder #{reminder_id}")
            return f"Cancelled reminder #{reminder_id}."
        raise ToolError("reminders action must be add, list or cancel")


# --------------------------------------------------------------------------- #
# Home Assistant
# --------------------------------------------------------------------------- #
class HomeAssistant:
    def __init__(self, cfg: HomeAssistantConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._client = client

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    def _require(self) -> tuple[str, dict[str, str]]:
        if not self.cfg.url:
            raise ToolError("Home Assistant is not configured (tools.home_assistant.url)")
        if not self.cfg.token:
            raise ToolError("Home Assistant is not configured (HOME_ASSISTANT_TOKEN is unset)")
        return self.cfg.url.rstrip("/"), {"Authorization": f"Bearer {self.cfg.token}"}

    async def run(self, args: dict[str, Any]) -> str:
        action = _str_arg(args, "action", max_len=20).lower()
        base, headers = self._require()
        if action == "list_states":
            return await self._list_states(base, headers, _str_arg(args, "domain", required=False, max_len=30))
        if action in ("turn_on", "turn_off", "toggle"):
            entity = _str_arg(args, "entity_id", max_len=100)
            domain = entity.split(".", 1)[0]
            if "." not in entity:
                raise ToolError(f"entity_id {entity!r} must look like light.kitchen")
            if domain not in self.cfg.allow_domains:
                raise ToolError(f"domain {domain!r} is not allowed; allowed: {', '.join(self.cfg.allow_domains)}")
            service = "turn_on" if domain in ("scene", "script") and action == "toggle" else action
            return await self._call_service(base, headers, domain, service, entity)
        raise ToolError("home_assistant action must be list_states, turn_on, turn_off or toggle")

    async def _list_states(self, base: str, headers: dict[str, str], domain: str | None) -> str:
        response = await self.client().get(f"{base}/api/states", headers=headers, timeout=self.cfg.timeout_s)
        if response.status_code == 401:
            raise ToolError("Home Assistant rejected the token")
        if response.status_code >= 400:
            raise ToolError(f"Home Assistant returned HTTP {response.status_code}")
        allowed = {domain} if domain else set(self.cfg.allow_domains)
        lines = []
        for state in response.json():
            entity_id = state.get("entity_id", "")
            if entity_id.split(".", 1)[0] not in allowed:
                continue
            name = (state.get("attributes") or {}).get("friendly_name") or entity_id
            lines.append(f"- {entity_id}: {state.get('state')} ({name})")
            if len(lines) >= 40:
                break
        return "\n".join(lines) if lines else "No matching entities."

    async def _call_service(self, base: str, headers: dict[str, str], domain: str, service: str, entity: str) -> str:
        response = await self.client().post(
            f"{base}/api/services/{domain}/{service}",
            headers=headers,
            json={"entity_id": entity},
            timeout=self.cfg.timeout_s,
        )
        if response.status_code == 401:
            raise ToolError("Home Assistant rejected the token")
        if response.status_code >= 400:
            raise ToolError(f"Home Assistant returned HTTP {response.status_code}")
        return f"{service} sent to {entity}."


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
class Shell:
    def __init__(self, cfg: ShellConfig) -> None:
        self.cfg = cfg

    async def run(self, args: dict[str, Any]) -> str:
        if not self.cfg.enabled:
            raise ToolError("shell is disabled")
        command = _str_arg(args, "command", max_len=300)
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"cannot parse command: {exc}") from exc
        if not argv:
            raise ToolError("empty command")
        if argv[0] not in self.cfg.whitelist:
            raise ToolError(f"{argv[0]!r} is not whitelisted; allowed: {', '.join(self.cfg.whitelist)}")
        for argument in argv[1:]:
            if argument.startswith("/") or ".." in argument:
                raise ToolError("absolute and parent paths are not allowed")
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise ToolError(f"{argv[0]!r} is not installed") from None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self.cfg.timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ToolError(f"command timed out after {self.cfg.timeout_s}s") from None
        output = stdout.decode("utf-8", "replace").strip()
        status = f"(exit {process.returncode})"
        return _cap(f"{output}\n{status}" if output else status, self.cfg.max_output_bytes)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]
    risky: bool = False

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def build_registry(
    cfg: ToolsConfig,
    *,
    http_client: httpx.AsyncClient | None = None,
    ddgs_search: Callable[[str, int], list[dict[str, Any]]] | None = None,
    reminder_store: ReminderStore | None = None,
    clock: Callable[[], datetime] | None = None,
) -> "ToolRegistry":
    return ToolRegistry(
        cfg,
        http_client=http_client,
        ddgs_search=ddgs_search,
        reminder_store=reminder_store,
        clock=clock,
    )


class ToolRegistry:
    def __init__(
        self,
        cfg: ToolsConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        ddgs_search: Callable[[str, int], list[dict[str, Any]]] | None = None,
        reminder_store: ReminderStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cfg = cfg
        self.http_client = http_client
        self.reminders = reminder_store or ReminderStore(cfg.reminders, clock=clock)
        self._tools: dict[str, ToolSpec] = {}
        builders: dict[str, Callable[[], ToolSpec]] = {
            "web_search": lambda: self._web_search(ddgs_search),
            "notes": self._notes,
            "reminders": self._reminders,
            "home_assistant": self._home_assistant,
            "shell": self._shell,
        }
        for name in cfg.enabled:
            self._tools[name] = builders[name]()

    # -- specs ----------------------------------------------------------- #
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self._tools.values()]

    def risky(self, name: str, args: dict[str, Any]) -> bool:
        if name not in self._tools:
            return False
        action = str(args.get("action", "")).lower()
        if name == "notes":
            return action in {"write", "append"}
        if name == "home_assistant":
            return action in {"turn_on", "turn_off", "toggle"}
        return self._tools[name].risky

    def describe(self, name: str, args: dict[str, Any]) -> str:
        if name == "notes":
            action = str(args.get("action", "")).lower()
            filename = args.get("filename") or "note"
            return f"{action} note {filename!r}"
        if name == "home_assistant":
            return f"{args.get('action', 'control')} {args.get('entity_id', 'devices')}"
        if name == "reminders":
            return f"{args.get('action', 'change')} reminder"
        if name == "shell":
            return f"run {args.get('command', 'a command')!r}"
        if name == "web_search":
            return f"search the web for {args.get('query', '')!r}"
        return name

    # -- execution ------------------------------------------------------- #
    async def call(self, name: str, args: dict[str, Any]) -> str:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"tool {name!r} is not available")
        if not isinstance(args, dict):
            raise ToolError(f"arguments for {name!r} must be an object")
        try:
            return await spec.handler(args)
        except ToolError:
            raise
        except Exception as exc:  # never leak a raw traceback into the conversation
            log.warning("tool %s failed: %s", name, exc)
            raise ToolError(f"{name} failed: {exc}") from exc

    def close(self) -> None:
        self.reminders.close()

    # -- tool builders --------------------------------------------------- #
    def _web_search(self, ddgs_search) -> ToolSpec:
        tool = WebSearch(self.cfg.web_search, client=self.http_client, ddgs_search=ddgs_search)
        return ToolSpec(
            name="web_search",
            description="Search the web and return the top results with short snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            handler=tool.run,
        )

    def _notes(self) -> ToolSpec:
        tool = Notes(self.cfg.notes)
        return ToolSpec(
            name="notes",
            description="Read and write the user's markdown notes.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "read", "write", "append"]},
                    "filename": {"type": "string", "description": "Note name, with or without .md"},
                    "content": {"type": "string", "description": "Markdown content for write/append"},
                },
                "required": ["action"],
            },
            handler=tool.run,
            risky=False,
        )

    def _reminders(self) -> ToolSpec:
        tool = Reminders(self.reminders)
        return ToolSpec(
            name="reminders",
            description="Add, list or cancel spoken reminders.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "cancel"]},
                    "text": {"type": "string", "description": "What to remind the user about"},
                    "when": {
                        "type": "string",
                        "description": "When: 'in 10 minutes', '2h30m', '7pm', 'tomorrow 9am' or ISO time",
                    },
                    "id": {"type": "integer", "description": "Reminder id for cancel"},
                },
                "required": ["action"],
            },
            handler=tool.run,
        )

    def _home_assistant(self) -> ToolSpec:
        tool = HomeAssistant(self.cfg.home_assistant, client=self.http_client)
        return ToolSpec(
            name="home_assistant",
            description="Read entity states or control Home Assistant devices.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list_states", "turn_on", "turn_off", "toggle"]},
                    "entity_id": {"type": "string", "description": "e.g. light.kitchen"},
                    "domain": {"type": "string", "description": "Filter for list_states, e.g. light"},
                },
                "required": ["action"],
            },
            handler=tool.run,
            risky=False,
        )

    def _shell(self) -> ToolSpec:
        tool = Shell(self.cfg.shell)
        return ToolSpec(
            name="shell",
            description="Run one read-only command from the user's whitelist and return its output.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Command line, e.g. 'df -h'"}},
                "required": ["command"],
            },
            handler=tool.run,
        )
