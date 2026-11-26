"""Tests for memory_cli.py: rendering, subcommands and exit codes."""

from __future__ import annotations

import pytest

from memory import FactStore
from memory_cli import main, render


@pytest.fixture()
def store(tmp_path) -> FactStore:
    return FactStore(tmp_path / "facts.db", seeds={"name": "User", "tone": "dry"})


def test_render_marks_stale(store):
    store.decay()
    fact = store.get("name")
    assert render(fact) in {"name=User", "name=User (stale)"}


def test_list_prints_every_fact(store, capsys):
    assert main(["list"], store=store) == 0
    out = capsys.readouterr().out
    assert "name=User" in out and "tone=dry" in out


def test_list_empty(tmp_path, capsys):
    empty = FactStore(tmp_path / "empty.db", seeds={})
    assert main(["list"], store=empty) == 0
    assert capsys.readouterr().out.strip() == "No facts stored."


def test_get_found_and_missing(store, capsys):
    assert main(["get", "name"], store=store) == 0
    assert "name=User" in capsys.readouterr().out
    assert main(["get", "nope"], store=store) == 1
    assert "not found" in capsys.readouterr().err


def test_set_then_get(store, capsys):
    assert main(["set", "ui_theme", "dark"], store=store) == 0
    assert "ui_theme=dark" in capsys.readouterr().out
    assert store.get("ui_theme").value == "dark"


def test_set_rejects_bad_key(store, capsys):
    assert main(["set", "Bad Key", "x"], store=store) == 2
    assert "invalid fact key" in capsys.readouterr().err


def test_forget(store, capsys):
    assert main(["forget", "tone"], store=store) == 0
    assert "forgot tone" in capsys.readouterr().out
    assert store.get("tone") is None
    assert main(["forget", "tone"], store=store) == 1


def test_decay(store, capsys):
    assert main(["decay"], store=store) == 0
    assert "marked stale" in capsys.readouterr().out
