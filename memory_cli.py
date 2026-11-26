"""List and edit Jarvis's remembered facts without touching SQLite by hand.

    python memory_cli.py list
    python memory_cli.py get ui_theme
    python memory_cli.py set ui_theme dark
    python memory_cli.py forget ui_theme
    python memory_cli.py decay

Uses the same database and seeds as `main.py`, so changes are picked up on the
next conversation. Exit codes: 0 ok, 1 not found, 2 bad config or value.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from config import Config, ConfigError, load_config
from memory import Fact, FactStore


def build_store(cfg: Config) -> FactStore:
    return FactStore(cfg.memory.sqlite_path, seeds=cfg.memory.seed_facts, decay_days=cfg.memory.decay_days)


def render(fact: Fact) -> str:
    stale = " (stale)" if fact.stale else ""
    return f"{fact.key}={fact.value}{stale}"


def cmd_list(store: FactStore) -> int:
    facts = store.all()
    if not facts:
        print("No facts stored.")
        return 0
    for fact in facts:
        print(render(fact))
    return 0


def cmd_get(store: FactStore, key: str) -> int:
    fact = store.get(key)
    if fact is None:
        print(f"{key}: not found", file=sys.stderr)
        return 1
    print(render(fact))
    return 0


def cmd_set(store: FactStore, key: str, value: str) -> int:
    try:
        fact = store.upsert(key, value)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(render(fact))
    return 0


def cmd_forget(store: FactStore, key: str) -> int:
    if not store.delete(key):
        print(f"{key}: not found", file=sys.stderr)
        return 1
    print(f"forgot {key}")
    return 0


def cmd_decay(store: FactStore) -> int:
    count = store.decay()
    print(f"{count} fact(s) marked stale.")
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and edit Jarvis's facts")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show every stored fact")
    get_parser = commands.add_parser("get", help="show one fact")
    get_parser.add_argument("key")
    set_parser = commands.add_parser("set", help="create or replace a fact")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    forget_parser = commands.add_parser("forget", help="delete a fact")
    forget_parser.add_argument("key")
    commands.add_parser("decay", help="flag facts unused for the decay window")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None, *, store: FactStore | None = None) -> int:
    args = parse_args(argv)
    owned = store is None
    if owned:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        store = build_store(cfg)
    try:
        if args.command == "list":
            return cmd_list(store)
        if args.command == "get":
            return cmd_get(store, args.key)
        if args.command == "set":
            return cmd_set(store, args.key, args.value)
        if args.command == "forget":
            return cmd_forget(store, args.key)
        if args.command == "decay":
            return cmd_decay(store)
    finally:
        if owned:
            store.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
