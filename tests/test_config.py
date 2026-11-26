"""Tests for config.py - the validation contract every other module relies on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config as jarvis_config
from config import ConfigError, load_config, redacted_config

REPO = Path(__file__).resolve().parents[1]

SEED_FACTS = {
    "name": "User",
    "timezone": "America/New_York",
    "preferred_language": "English",
    "preferred_tts_voice": "en_US-lessac-medium",
    "home_assistant_url": "http://localhost:8123",
    "assistant_name": "Jarvis",
    "tone": "concise, slightly dry humor, like a competent butler",
}

MINIMAL = "version: 1\naudio:\n  wakeword:\n    enabled: false\n"


@pytest.fixture(autouse=True)
def _clear_cache():
    jarvis_config.reset_cache()
    yield
    jarvis_config.reset_cache()


def write_config(directory: Path, body: str = MINIMAL, name: str = "config.yaml") -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the shipped file
# --------------------------------------------------------------------------- #
def test_shipped_config_loads_with_spec_seed_facts():
    cfg = load_config(REPO / "config.yaml", reload=True)

    assert cfg.version == 1
    assert cfg.memory.seed_facts == SEED_FACTS
    assert cfg.memory.decay_days == 90
    assert cfg.assistant.name == "Jarvis"


def test_shipped_config_paths_are_absolute_and_tts_path_is_derived():
    cfg = load_config(REPO / "config.yaml", reload=True)

    for path in (
        cfg.memory.sqlite_path,
        cfg.memory.chroma.path,
        cfg.assistant.data_dir,
        cfg.tts.voices_dir,
    ):
        assert path.is_absolute()
    assert cfg.tts.model_path is None
    assert cfg.tts_model_path() == cfg.tts.voices_dir / "en_US-lessac-medium.onnx"


def test_shipped_config_routing_order_is_local_first():
    cfg = load_config(REPO / "config.yaml", reload=True)

    assert [p.name for p in cfg.routing_order()] == ["ollama", "openai", "anthropic"]
    assert cfg.tools.shell.whitelist == ["ls", "date", "uptime", "df", "whoami", "pwd"]


# --------------------------------------------------------------------------- #
# env:NAME indirection
# --------------------------------------------------------------------------- #
def test_env_ref_resolves_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_KEY", "sk-not-a-real-key")
    path = write_config(
        tmp_path,
        "version: 1\nllm:\n  providers:\n    - name: openai\n      model: gpt-4o-mini\n"
        "      api_key: env:JARVIS_TEST_KEY\n",
    )

    cfg = load_config(path)

    assert cfg.provider("openai").api_key == "sk-not-a-real-key"
    assert cfg.provider("openai").usable is True


def test_missing_env_var_becomes_null_plus_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_ABSENT_KEY", raising=False)
    path = write_config(
        tmp_path,
        "version: 1\nllm:\n  providers:\n    - name: openai\n      model: gpt-4o-mini\n"
        "      api_key: env:JARVIS_ABSENT_KEY\n",
    )

    cfg = load_config(path)

    assert cfg.provider("openai").api_key is None
    assert cfg.provider("openai").usable is False
    assert any("JARVIS_ABSENT_KEY" in warning for warning in cfg.warnings)


def test_dollar_brace_env_syntax_also_works(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_TOKEN", "token-value")
    path = write_config(
        tmp_path,
        "version: 1\ntools:\n  home_assistant:\n    token: ${JARVIS_TEST_TOKEN}\n",
    )

    assert load_config(path).tools.home_assistant.token == "token-value"


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def test_relative_paths_resolve_against_config_dir_not_cwd(tmp_path, monkeypatch):
    path = write_config(
        tmp_path,
        "version: 1\nmemory:\n  sqlite_path: data/jarvis.db\n  chroma:\n    path: data/chroma\n",
    )
    monkeypatch.chdir(REPO)

    cfg = load_config(path)

    assert cfg.memory.sqlite_path == tmp_path / "data" / "jarvis.db"
    assert cfg.memory.chroma.path == tmp_path / "data" / "chroma"


def test_tts_model_path_override_wins(tmp_path):
    path = write_config(tmp_path, "version: 1\ntts:\n  model_path: voices/custom.onnx\n")

    cfg = load_config(path)

    assert cfg.tts_model_path() == tmp_path / "voices" / "custom.onnx"


# --------------------------------------------------------------------------- #
# validation failures are loud and readable
# --------------------------------------------------------------------------- #
def test_unknown_key_is_rejected(tmp_path):
    path = write_config(tmp_path, "version: 1\nassistant:\n  nam: Jarvis\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "assistant.nam" in message
    assert "invalid configuration" in message


def test_wakeword_requires_16khz(tmp_path):
    path = write_config(
        tmp_path,
        "version: 1\naudio:\n  sample_rate: 8000\n  wakeword:\n    enabled: true\n",
    )

    with pytest.raises(ConfigError, match="16000"):
        load_config(path)


def test_non_80ms_frames_warn_instead_of_failing(tmp_path):
    path = write_config(
        tmp_path,
        "version: 1\naudio:\n  frame_ms: 20\n  wakeword:\n    enabled: true\n",
    )

    cfg = load_config(path)

    assert cfg.audio.frame_ms == 20
    assert any("80 ms" in warning for warning in cfg.warnings)


@pytest.mark.parametrize("entry", ["df -h", "/bin/ls", "ls; rm -rf /", ".."])
def test_shell_whitelist_rejects_anything_but_bare_command_names(tmp_path, entry):
    path = write_config(
        tmp_path,
        f"version: 1\ntools:\n  shell:\n    enabled: true\n    whitelist: ['{entry}']\n",
    )

    with pytest.raises(ConfigError, match="whitelist"):
        load_config(path)


def test_duplicate_provider_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        "version: 1\nllm:\n  providers:\n"
        "    - name: ollama\n      model: a\n"
        "    - name: ollama\n      model: b\n",
    )

    with pytest.raises(ConfigError, match="duplicate"):
        load_config(path)


def test_future_config_version_is_rejected(tmp_path):
    path = write_config(tmp_path, "version: 99\n")

    with pytest.raises(ConfigError, match="version"):
        load_config(path)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_broken_yaml_raises_config_error(tmp_path):
    path = write_config(tmp_path, "version: 1\nassistant: [unclosed\n")

    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


# --------------------------------------------------------------------------- #
# secrets never reach the report
# --------------------------------------------------------------------------- #
def test_redacted_config_hides_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_KEY", "sk-live-do-not-print")
    path = write_config(
        tmp_path,
        "version: 1\nllm:\n  providers:\n    - name: openai\n      model: gpt-4o-mini\n"
        "      api_key: env:JARVIS_TEST_KEY\n",
    )

    cfg = load_config(path)
    dumped = json.dumps(redacted_config(cfg))

    assert "sk-live-do-not-print" not in dumped
    assert '"<set>"' in dumped
    # Compare against the JSON-escaped form: on Windows every "\" becomes "\\".
    assert json.dumps(str(tmp_path))[1:-1] in dumped


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def test_local_only_routing_drops_cloud_providers(tmp_path):
    path = write_config(
        tmp_path,
        "version: 1\nllm:\n  routing:\n    mode: local_only\n  providers:\n"
        "    - name: ollama\n      model: llama3.1:8b\n"
        "    - name: openai\n      model: gpt-4o-mini\n      api_key: sk-x\n",
    )

    assert [p.name for p in load_config(path).routing_order()] == ["ollama"]


def test_ensure_runtime_dirs_creates_missing_directories(tmp_path):
    path = write_config(
        tmp_path,
        "version: 1\nassistant:\n  data_dir: state\n  log_dir: state/logs\n"
        "  prompts_dir: state/prompts\nmemory:\n  sqlite_path: state/jarvis.db\n"
        "  chroma:\n    path: state/chroma\n",
    )

    cfg = load_config(path)
    created = cfg.ensure_runtime_dirs()

    assert (tmp_path / "state" / "chroma").is_dir()
    assert (tmp_path / "state" / "logs").is_dir()
    assert any(path.name == "chroma" for path in created)
    assert cfg.ensure_runtime_dirs() == []  # idempotent
