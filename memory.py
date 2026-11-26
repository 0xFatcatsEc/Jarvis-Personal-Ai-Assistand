"""Memory: SQLite user facts plus Chroma conversation recall.

Three jobs, matching the spec:

* store  - `user_facts` table, pre-seeded once from `memory.seed_facts`
* recall - fetch core facts plus facts matching the current topic, and pull
           past exchanges out of Chroma by embedding similarity
* learn  - after each exchange, ask the LLM for new user-stated facts and
           upsert them (credentials and one-off remarks are refused)

Facts unaccessed for `decay_days` are flagged `stale`; prompts then tell the
LLM to confirm them instead of asserting them.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Protocol, Sequence

from config import ChromaConfig, MemoryConfig

log = logging.getLogger(__name__)

CORE_KEYS = ("name", "timezone", "tone", "assistant_name")
STOPWORDS = frozenset(
    "a an and are as at be but by do does for from had has have how i if in is it its me my "
    "of on or our so that the their them then there these they this to was we were what when "
    "where which who why will with you your".split()
)
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
SENSITIVE_RE = re.compile(
    r"password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|"
    r"credit[_-]?card|card[_-]?number|cvv|ssn|social[_-]?security|bank|routing[_-]?number|"
    r"pin[_-]?code|seed[_-]?phrase",
    re.IGNORECASE,
)

EXTRACTION_PROMPT = """\
Extract durable facts about the user from the exchange below.

Return ONLY a JSON array, no prose and no code fences, in this shape:
[{{"key": "ui_theme", "value": "dark"}}]

Rules:
- Only facts the user stated about themselves: preferences, corrections, \
stable personal details, names of people or places they mention about themselves.
- key is lowercase_snake_case, at most 48 characters, and names the attribute \
(ui_theme, not "user likes dark mode").
- Skip anything transient ("I'm tired today"), anything the assistant said, \
questions, and anything already known unchanged.
- Never store passwords, tokens, keys, payment or identity numbers. If the user \
offers one, omit it entirely.
- Return [] when there is nothing new.

Known facts:
{known}

Exchange:
User: {user}
Assistant: {assistant}
"""


# --------------------------------------------------------------------------- #
# facts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    created_at: str
    last_accessed: str | None = None
    stale: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat()


class FactStore:
    """The `user_facts` table. Transactions are small and local."""

    def __init__(
        self,
        db_path: Path,
        *,
        seeds: dict[str, str] | None = None,
        decay_days: int = 90,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.path = Path(db_path)
        self.seeds = dict(seeds or {})
        self.decay_days = decay_days
        self._clock = clock
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._seeded = False

    # -- plumbing -------------------------------------------------------- #
    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA foreign_keys=ON")
            self._db = db
            self._migrate(db)
        return self._db

    def _ensure_seeded(self) -> None:
        if not self._seeded:
            self._insert_seeds(self._connection())
            self._seeded = True

    def _insert_seeds(self, db: sqlite3.Connection) -> int:
        """INSERT OR IGNORE every seed; returns the number actually added."""
        now = _iso(self._clock())
        added = 0
        for key, value in self.seeds.items():
            cursor = db.execute(
                "INSERT OR IGNORE INTO user_facts (key, value, created_at) VALUES (?, ?, ?)",
                (key, str(value), now),
            )
            added += cursor.rowcount
        db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('seeded_at', ?)", (now,))
        db.commit()
        return added

    def _migrate(self, db: sqlite3.Connection) -> None:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_facts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                key           TEXT NOT NULL UNIQUE,
                value         TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                last_accessed TEXT,
                stale         INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(user_facts)")}
        if "stale" not in columns:  # database from an older build
            db.execute("ALTER TABLE user_facts ADD COLUMN stale INTEGER NOT NULL DEFAULT 0")
        db.commit()

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    # -- reads ----------------------------------------------------------- #
    def _row(self, row: sqlite3.Row) -> Fact:
        return Fact(
            key=row["key"],
            value=row["value"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            stale=bool(row["stale"]),
        )

    def all(self) -> list[Fact]:
        with self._lock:
            self._ensure_seeded()
            rows = self._connection().execute("SELECT * FROM user_facts ORDER BY key").fetchall()
        return [self._row(row) for row in rows]

    def get(self, key: str) -> Fact | None:
        with self._lock:
            self._ensure_seeded()
            row = self._connection().execute("SELECT * FROM user_facts WHERE key = ?", (key,)).fetchone()
        return self._row(row) if row else None

    # -- writes ---------------------------------------------------------- #
    def seed(self) -> int:
        """Insert seed facts that do not exist yet. Returns how many were added."""
        with self._lock:
            if self._seeded:
                return 0
            added = self._insert_seeds(self._connection())
            self._seeded = True
        if added:
            log.info("seeded %d user fact(s)", added)
        return added

    def upsert(self, key: str, value: str) -> Fact:
        """Set a fact's value. `created_at` survives updates; staleness clears."""
        key = key.strip()
        value = str(value).strip()
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"invalid fact key {key!r}; use lowercase_snake_case")
        if not value:
            raise ValueError(f"fact {key!r} has an empty value")
        now = _iso(self._clock())
        with self._lock:
            db = self._connection()
            self._ensure_seeded()
            db.execute(
                """
                INSERT INTO user_facts (key, value, created_at, last_accessed, stale)
                VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    last_accessed = excluded.last_accessed,
                    stale = 0
                """,
                (key, value, now, now),
            )
            db.commit()
            row = db.execute("SELECT * FROM user_facts WHERE key = ?", (key,)).fetchone()
        return self._row(row)

    def delete(self, key: str) -> bool:
        """Remove a fact. Returns True if it existed."""
        with self._lock:
            db = self._connection()
            self._ensure_seeded()
            cursor = db.execute("DELETE FROM user_facts WHERE key = ?", (key.strip(),))
            db.commit()
        return cursor.rowcount > 0

    def touch(self, keys: Iterable[str]) -> int:
        keys = [key for key in keys if key]
        if not keys:
            return 0
        now = _iso(self._clock())
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            db = self._connection()
            self._ensure_seeded()
            cursor = db.execute(
                f"UPDATE user_facts SET last_accessed = ?, stale = 0 WHERE key IN ({placeholders})",
                [now, *keys],
            )
            db.commit()
        return cursor.rowcount

    def decay(self) -> int:
        """Flag facts unused for `decay_days`. Returns newly stale count."""
        cutoff = _iso(self._clock() - timedelta(days=self.decay_days))
        with self._lock:
            db = self._connection()
            self._ensure_seeded()
            cursor = db.execute(
                """
                UPDATE user_facts SET stale = 1
                WHERE stale = 0 AND COALESCE(last_accessed, created_at) < ?
                """,
                (cutoff,),
            )
            db.commit()
        if cursor.rowcount:
            log.info("%d fact(s) decayed to stale", cursor.rowcount)
        return cursor.rowcount

    # -- recall ---------------------------------------------------------- #
    def recall(self, topic: str, *, top_k: int = 8) -> list[Fact]:
        """Core facts plus the best topic matches, most relevant first."""
        facts = {fact.key: fact for fact in self.all()}
        picked: list[Fact] = [facts[key] for key in CORE_KEYS if key in facts]

        tokens = _tokens(topic)
        scored: list[tuple[float, Fact]] = []
        for key, fact in facts.items():
            if key in CORE_KEYS:
                continue
            score = _score(fact, tokens, topic)
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda item: (-item[0], item[1].key))

        for _, fact in scored:
            if len(picked) >= top_k:
                break
            picked.append(fact)
        picked = picked[:top_k]
        self.touch(fact.key for fact in picked)
        return picked


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9']+", text.lower())
        if len(word) >= 2 and word not in STOPWORDS
    }


def _score(fact: Fact, tokens: set[str], topic: str) -> float:
    if not tokens:
        return 0.0
    key_tokens = _tokens(fact.key.replace("_", " "))
    value_tokens = _tokens(fact.value)
    score = 3.0 * len(tokens & key_tokens) + 1.0 * len(tokens & value_tokens)
    phrase = topic.strip().lower()
    if phrase and phrase in fact.value.lower():
        score += 2.0
    return score


def format_facts(facts: Sequence[Fact]) -> str:
    """`name=User, timezone=America/New_York (stale: confirm if relevant)`."""
    parts = []
    for fact in facts:
        suffix = " (stale: confirm if relevant)" if fact.stale else ""
        parts.append(f"{fact.key}={fact.value}{suffix}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# conversation recall
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str
    ts: str = ""
    similarity: float | None = None

    def text(self, *, limit: int = 400) -> str:
        return f"User: {_clip(self.user, limit)} -> Assistant: {_clip(self.assistant, limit)}"


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class ConversationStore(Protocol):
    def add_turn(self, turn: Turn) -> None: ...

    def search(self, query: str, *, top_k: int, min_similarity: float) -> list[Turn]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class NullConversations:
    """Used when Chroma is unavailable: facts still work, recall does not."""

    def add_turn(self, turn: Turn) -> None:
        return None

    def search(self, query: str, *, top_k: int, min_similarity: float) -> list[Turn]:
        return []

    def count(self) -> int:
        return 0

    def close(self) -> None:
        return None


class ChromaConversations:
    """Chroma persistent collection of past exchanges."""

    def __init__(self, cfg: ChromaConfig) -> None:
        self.cfg = cfg
        self._client: object | None = None
        self._collection: object | None = None

    def _collection_handle(self) -> object:
        if self._collection is None:
            import chromadb

            path = Path(self.cfg.path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
            self._collection = self._client.get_or_create_collection(  # type: ignore[attr-defined]
                name=self.cfg.collection,
                embedding_function=_embedding_function(self.cfg.embedding_model),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_turn(self, turn: Turn) -> None:
        collection = self._collection_handle()
        turn_id = f"{turn.ts or _iso(_utcnow())}-{abs(hash(turn.user + turn.assistant)) % 10**8}"
        collection.add(  # type: ignore[attr-defined]
            ids=[turn_id],
            documents=[turn.text()],
            metadatas=[{"ts": turn.ts, "user": _clip(turn.user, 500)}],
        )

    def search(self, query: str, *, top_k: int, min_similarity: float) -> list[Turn]:
        if top_k <= 0 or not query.strip():
            return []
        collection = self._collection_handle()
        if collection.count() == 0:  # type: ignore[attr-defined]
            return []
        result = collection.query(  # type: ignore[attr-defined]
            query_texts=[query],
            n_results=min(top_k, collection.count()),  # type: ignore[attr-defined]
            include=["documents", "metadatas", "distances"],
        )
        turns: list[Turn] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            similarity = 1.0 - float(distance)
            if similarity < min_similarity:
                continue
            user, assistant = _split_document(document)
            turns.append(
                Turn(
                    user=user,
                    assistant=assistant,
                    ts=str((metadata or {}).get("ts", "")),
                    similarity=round(similarity, 3),
                )
            )
        return turns

    def count(self) -> int:
        return int(self._collection_handle().count())  # type: ignore[attr-defined]

    def close(self) -> None:
        self._collection = None
        self._client = None


def _embedding_function(model: str):
    import chromadb.utils.embedding_functions as ef

    default = "all-MiniLM-L6-v2"
    if model in (default, "default", ""):
        return ef.DefaultEmbeddingFunction()
    return ef.SentenceTransformerEmbeddingFunction(model_name=model)


def _split_document(document: str) -> tuple[str, str]:
    user, _, rest = document.partition("\nAssistant: ")
    if rest:
        return user.removeprefix("User: "), rest
    return document.removeprefix("User: "), ""


# --------------------------------------------------------------------------- #
# learning
# --------------------------------------------------------------------------- #
def parse_facts(
    raw: str,
    *,
    max_new: int = 5,
    known: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Parse the extractor's JSON into validated, de-duplicated (key, value)."""
    text = raw.strip()
    if "[" in text and "]" in text:
        text = text[text.index("[") : text.rindex("]") + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("facts", [])
    if not isinstance(data, list):
        return []

    known = known or {}
    found: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip().lower().replace(" ", "_")
        value = str(item.get("value", "")).strip()
        if not key or not value or not KEY_RE.fullmatch(key):
            continue
        if SENSITIVE_RE.search(key) or SENSITIVE_RE.search(value):
            log.info("refusing to store sensitive fact %r", key)
            continue
        if known.get(key) == value:
            continue
        found[key] = value
        if len(found) >= max_new:
            break
    return list(found.items())


@dataclass
class RecallContext:
    facts: list[Fact] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)

    def facts_text(self) -> str:
        return format_facts(self.facts)

    def memory_lines(self) -> list[str]:
        return [turn.text() for turn in self.turns]


class Memory:
    """Facade over FactStore + ConversationStore used by the main loop."""

    def __init__(
        self,
        cfg: MemoryConfig,
        *,
        store: FactStore | None = None,
        conversations: ConversationStore | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.cfg = cfg
        self.store = store or FactStore(
            cfg.sqlite_path, seeds=cfg.seed_facts, decay_days=cfg.decay_days, clock=clock
        )
        self.store.seed()
        if conversations is not None:
            self.conversations = conversations
        else:
            try:
                self.conversations = ChromaConversations(cfg.chroma)
            except Exception as exc:  # chroma missing/broken: degrade, do not die
                log.warning("conversation recall disabled: %s", exc)
                self.conversations = NullConversations()

    # -- facts ----------------------------------------------------------- #
    def facts(self) -> list[Fact]:
        return self.store.all()

    def get(self, key: str) -> Fact | None:
        return self.store.get(key)

    def set_fact(self, key: str, value: str) -> Fact:
        return self.store.upsert(key, value)

    def forget(self, key: str) -> bool:
        return self.store.delete(key)

    def recall_context(self, query: str) -> RecallContext:
        facts = self.store.recall(query, top_k=self.cfg.recall.top_k_facts)
        try:
            turns = self.conversations.search(
                query,
                top_k=self.cfg.recall.top_k_turns,
                min_similarity=self.cfg.recall.min_similarity,
            )
        except Exception as exc:  # never let recall break a reply
            log.warning("conversation recall failed: %s", exc)
            turns = []
        return RecallContext(facts=facts, turns=turns)

    # -- conversation ---------------------------------------------------- #
    def remember_turn(self, user: str, assistant: str, *, ts: str | None = None) -> None:
        try:
            self.conversations.add_turn(Turn(user=user, assistant=assistant, ts=ts or _iso(_utcnow())))
        except Exception as exc:
            log.warning("could not store conversation turn: %s", exc)

    # -- learning -------------------------------------------------------- #
    async def learn(
        self,
        user_text: str,
        assistant_text: str,
        llm: Callable[[str], Awaitable[str]],
    ) -> list[Fact]:
        """Extract new user facts with `llm(prompt) -> raw text` and store them."""
        if not self.cfg.learn.enabled or not user_text.strip():
            return []
        known = {fact.key: fact.value for fact in self.store.all()}
        prompt = EXTRACTION_PROMPT.format(
            known=format_facts(self.store.all()) or "- (none)",
            user=_clip(user_text, 2000),
            assistant=_clip(assistant_text, 2000),
        )
        try:
            raw = await llm(prompt)
        except Exception as exc:
            log.warning("fact extraction failed: %s", exc)
            return []
        pairs = parse_facts(raw, max_new=self.cfg.learn.max_new_facts, known=known)
        stored = [self.store.upsert(key, value) for key, value in pairs]
        if stored:
            log.info("learned %d fact(s): %s", len(stored), ", ".join(f.key for f in stored))
        return stored

    # -- maintenance ----------------------------------------------------- #
    def decay(self) -> int:
        return self.store.decay()

    def close(self) -> None:
        self.store.close()
        self.conversations.close()
