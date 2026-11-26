"""Jarvis configuration: load, validate and expose config.yaml.

    python config.py --check          # validate ./config.yaml, print a summary
    python config.py --check --json   # same, as redacted JSON
    python config.py -p other.yaml    # validate a different file

Public API:
    load_config(path=None) -> Config   validated, cached per file
    get_config() -> Config             process-wide singleton
    ConfigError                        bad or missing configuration

Conventions honoured here (see config.yaml for the user-facing docs):
  * `env:NAME` / `${NAME}` values are resolved from the environment; a
    missing variable becomes None plus a warning, never a crash.
  * Relative paths are resolved against the config file's directory.
  * Unknown keys are rejected so typos fail loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

CONFIG_VERSION = 1
DEFAULT_CONFIG_FILE = "config.yaml"
CONFIG_ENV_VAR = "JARVIS_CONFIG"

# `env:OPENAI_API_KEY` or `${OPENAI_API_KEY}`
_ENV_REF = re.compile(
    r"^(?:env:(?P<plain>[A-Za-z_][A-Za-z0-9_]*)|\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\})$"
)
_SEED_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

# Dotted keys (no list indices needed yet) whose values are filesystem paths.
_PATH_KEYS: tuple[str, ...] = (
    "assistant.log_dir",
    "assistant.data_dir",
    "assistant.prompts_dir",
    "stt.download_root",
    "memory.sqlite_path",
    "memory.chroma.path",
    "tts.voices_dir",
    "tts.model_path",
    "tools.notes.dir",
    "tools.reminders.db_path",
)

FALLBACK_REASONS = frozenset(
    {"connection", "timeout", "rate_limit", "server_error", "bad_response"}
)
TOOL_NAMES = ("web_search", "notes", "reminders", "home_assistant", "shell")


class ConfigError(RuntimeError):
    """config.yaml is missing, unreadable, or fails validation."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
class AssistantConfig(_Section):
    name: str = Field("Jarvis", min_length=1, max_length=40)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("logs")
    data_dir: Path = Path("data")
    prompts_dir: Path = Path("prompts")


class WakeWordConfig(_Section):
    enabled: bool = True
    model: str = Field("alexa", min_length=1)
    threshold: float = Field(0.55, ge=0.0, le=1.0)
    patience: int = Field(2, ge=1, le=10)
    cooldown_s: float = Field(2.0, ge=0.0, le=60.0)


class VadConfig(_Section):
    energy_threshold: float = Field(0.012, gt=0.0, le=1.0)
    silence_ms: int = Field(800, ge=100, le=10_000)
    min_speech_ms: int = Field(300, ge=50, le=10_000)
    max_utterance_s: float = Field(15.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def _sane_windows(self) -> VadConfig:
        if self.min_speech_ms >= self.max_utterance_s * 1000:
            raise ValueError("audio.vad.min_speech_ms must be below max_utterance_s")
        return self


class AudioConfig(_Section):
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = Field(16_000, ge=8_000, le=48_000)
    frame_ms: Literal[10, 20, 30, 40, 60, 80] = 80
    channels: Literal[1] = 1
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VadConfig = Field(default_factory=VadConfig)


class SttConfig(_Section):
    engine: Literal["faster-whisper"] = "faster-whisper"
    model: str = Field("base.en", min_length=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: Literal["int8", "int8_float16", "float16", "float32"] = "int8"
    language: str | None = "en"
    beam_size: int = Field(5, ge=1, le=10)
    vad_filter: bool = True
    cpu_threads: int = Field(0, ge=0, le=64)
    download_root: Path = Path("models/whisper")


class ProviderConfig(_Section):
    name: Literal["ollama", "openai", "anthropic"]
    enabled: bool = True
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = Field(60.0, gt=0.0, le=600.0)
    max_retries: int = Field(2, ge=0, le=5)
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def needs_key(self) -> bool:
        return self.name in ("openai", "anthropic")

    @property
    def usable(self) -> bool:
        """Enabled and has what it needs to be called."""
        return self.enabled and (not self.needs_key or bool(self.api_key))

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is not None and not re.match(r"^https?://", value):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/") if value else value


class RoutingConfig(_Section):
    mode: Literal["sequential", "local_only", "cloud_only"] = "sequential"
    fallback_on: list[str] = Field(
        default_factory=lambda: ["connection", "timeout", "rate_limit", "server_error"]
    )

    @field_validator("fallback_on")
    @classmethod
    def _known_reasons(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - FALLBACK_REASONS)
        if unknown:
            raise ValueError(
                f"unknown fallback reason(s) {unknown}; allowed: {sorted(FALLBACK_REASONS)}"
            )
        return value


class LlmConfig(_Section):
    temperature: float = Field(0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=32, le=8192)
    history_turns: int = Field(8, ge=0, le=50)
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [ProviderConfig(name="ollama", model="llama3.1:8b")],
        min_length=1,
    )
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @model_validator(mode="after")
    def _providers_unique(self) -> LlmConfig:
        names = [p.name for p in self.providers]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate llm provider(s) {dupes}; list each provider once")
        if not any(p.enabled for p in self.providers):
            raise ValueError("every llm provider is disabled; enable at least one")
        return self


class ChromaConfig(_Section):
    path: Path = Path("data/chroma")
    collection: str = Field("conversations", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,62}$")
    embedding_model: str = "all-MiniLM-L6-v2"


class RecallConfig(_Section):
    top_k_facts: int = Field(8, ge=1, le=64)
    top_k_turns: int = Field(5, ge=0, le=50)
    min_similarity: float = Field(0.25, ge=0.0, le=1.0)


class LearnConfig(_Section):
    enabled: bool = True
    max_new_facts: int = Field(5, ge=1, le=20)
    max_tokens: int = Field(256, ge=64, le=2048)


class MemoryConfig(_Section):
    sqlite_path: Path = Path("data/jarvis.db")
    decay_days: int = Field(90, ge=1, le=3650)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    recall: RecallConfig = Field(default_factory=RecallConfig)
    learn: LearnConfig = Field(default_factory=LearnConfig)
    seed_facts: dict[str, str] = Field(default_factory=dict)

    @field_validator("seed_facts")
    @classmethod
    def _valid_seed_keys(cls, value: dict[str, str]) -> dict[str, str]:
        bad = sorted(k for k in value if not _SEED_KEY.match(k))
        if bad:
            raise ValueError(
                f"invalid seed fact key(s) {bad}; use lowercase_snake_case "
                "(e.g. ui_theme), since keys are upserted by name"
            )
        for key, fact in value.items():
            if not str(fact).strip():
                raise ValueError(f"seed fact {key!r} has an empty value")
        return {k: str(v) for k, v in value.items()}


class TtsConfig(_Section):
    engine: Literal["piper"] = "piper"
    voice: str = Field("en_US-lessac-medium", min_length=1)
    voices_dir: Path = Path("models/piper")
    model_path: Path | None = None
    sample_rate: int = Field(22_050, ge=8_000, le=48_000)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    volume: float = Field(1.0, ge=0.0, le=1.0)
    device_index: int | None = Field(None, ge=0)


class WebSearchConfig(_Section):
    provider: Literal["ddgs", "tavily", "searxng"] = "ddgs"
    max_results: int = Field(5, ge=1, le=10)
    timeout_s: float = Field(10.0, gt=0.0, le=60.0)
    api_key: str | None = None
    searxng_url: str | None = None

    @model_validator(mode="after")
    def _provider_requirements(self) -> WebSearchConfig:
        if self.provider == "searxng" and not self.searxng_url:
            raise ValueError("tools.web_search.searxng_url is required when provider: searxng")
        return self


class HomeAssistantConfig(_Section):
    url: str | None = None
    token: str | None = None
    timeout_s: float = Field(10.0, gt=0.0, le=60.0)
    allow_domains: list[str] = Field(
        default_factory=lambda: ["light", "switch", "climate", "scene", "script"]
    )


class NotesConfig(_Section):
    dir: Path = Path("notes")
    max_bytes: int = Field(64_000, ge=1_000, le=10_000_000)


class RemindersConfig(_Section):
    db_path: Path = Path("data/reminders.db")
    check_interval_s: int = Field(30, ge=5, le=3600)
    max_per_user: int = Field(20, ge=1, le=500)


class ShellConfig(_Section):
    enabled: bool = True
    whitelist: list[str] = Field(
        default_factory=lambda: ["ls", "date", "uptime", "df", "whoami", "pwd"]
    )
    timeout_s: float = Field(5.0, gt=0.0, le=60.0)
    max_output_bytes: int = Field(4096, ge=256, le=1_000_000)

    @field_validator("whitelist")
    @classmethod
    def _plain_command_names(cls, value: list[str]) -> list[str]:
        for entry in value:
            # argv[0] looked up on PATH: no paths, flags, spaces or shell syntax.
            if not re.fullmatch(r"[A-Za-z0-9._-]+", entry):
                raise ValueError(
                    f"invalid shell whitelist entry {entry!r}; use a bare command "
                    "name (e.g. 'df'), never a path, flag or shell expression"
                )
            if entry in (".", ".."):
                raise ValueError(f"invalid shell whitelist entry {entry!r}")
        return value


class ToolsConfig(_Section):
    enabled: list[str] = Field(default_factory=lambda: list(TOOL_NAMES))
    confirm_risky_actions: bool = True
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    home_assistant: HomeAssistantConfig = Field(default_factory=HomeAssistantConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    reminders: RemindersConfig = Field(default_factory=RemindersConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)

    @field_validator("enabled")
    @classmethod
    def _known_tools(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(TOOL_NAMES))
        if unknown:
            raise ValueError(f"unknown tool(s) {unknown}; known: {list(TOOL_NAMES)}")
        return value

    @model_validator(mode="after")
    def _shell_guard(self) -> ToolsConfig:
        if "shell" in self.enabled and self.shell.enabled and not self.shell.whitelist:
            raise ValueError("tools.shell.whitelist cannot be empty while shell is enabled")
        return self


# --------------------------------------------------------------------------- #
# root
# --------------------------------------------------------------------------- #
class Config(_Section):
    version: int = CONFIG_VERSION
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    source_path: Path | None = Field(None, exclude=True)
    warnings: tuple[str, ...] = Field((), exclude=True)

    @model_validator(mode="after")
    def _cross_checks(self) -> Config:
        if self.version != CONFIG_VERSION:
            raise ValueError(
                f"unsupported config version {self.version}; this build expects {CONFIG_VERSION}"
            )
        if self.audio.wakeword.enabled and self.audio.sample_rate != 16_000:
            raise ValueError(
                "audio.sample_rate must be 16000 while audio.wakeword.enabled "
                "(openWakeWord and Whisper are 16 kHz)"
            )
        return self

    # -- lookups used by brain/tools/tts -------------------------------- #
    def provider(self, name: str) -> ProviderConfig | None:
        for candidate in self.llm.providers:
            if candidate.name == name:
                return candidate
        return None

    def routing_order(self) -> list[ProviderConfig]:
        """Providers to try, in order, for the configured routing mode."""
        mode = self.llm.routing.mode
        if mode == "local_only":
            return [p for p in self.llm.providers if p.enabled and p.name == "ollama"]
        if mode == "cloud_only":
            return [p for p in self.llm.providers if p.enabled and p.name != "ollama"]
        return [p for p in self.llm.providers if p.enabled]

    def tts_model_path(self) -> Path:
        if self.tts.model_path is not None:
            return self.tts.model_path
        return self.tts.voices_dir / f"{self.tts.voice}.onnx"

    def ensure_runtime_dirs(self) -> list[Path]:
        """Create the writable directories Jarvis needs. Returns what it made."""
        wanted = {
            self.assistant.data_dir,
            self.assistant.log_dir,
            self.assistant.prompts_dir,
            self.tools.notes.dir,
            self.memory.chroma.path,
            self.memory.sqlite_path.parent,
            self.tools.reminders.db_path.parent,
        }
        created: list[Path] = []
        for directory in sorted(wanted, key=str):
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)
        return created


# --------------------------------------------------------------------------- #
# loading pipeline: yaml -> env refs -> paths -> validate -> warnings
# --------------------------------------------------------------------------- #
def _config_path(path: str | Path | None) -> Path:
    raw = path if path is not None else os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_FILE
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.normpath(candidate))


def _resolve_env_refs(node: Any, *, missing: list[str], trail: str = "") -> Any:
    if isinstance(node, dict):
        return {
            key: _resolve_env_refs(
                value, missing=missing, trail=f"{trail}.{key}" if trail else str(key)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            _resolve_env_refs(value, missing=missing, trail=f"{trail}[{index}]")
            for index, value in enumerate(node)
        ]
    if isinstance(node, str):
        match = _ENV_REF.match(node.strip())
        if match:
            name = match.group("plain") or match.group("braced")
            value = os.environ.get(name)
            if not value:
                missing.append(f"{trail} -> ${name}")
                return None
            return value
    return node


def _absolutize(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.normpath(path))


def _resolve_paths(data: dict[str, Any], base: Path) -> None:
    for dotted in _PATH_KEYS:
        *parents, leaf = dotted.split(".")
        node: Any = data
        for part in parents:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
        if not isinstance(node, dict):
            continue
        value = node.get(leaf)
        if isinstance(value, str) and value.strip():
            node[leaf] = _absolutize(value, base)


def _format_validation_error(exc: ValidationError, path: Path) -> str:
    lines = [f"invalid configuration in {path}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def _optional_env_ref(entry: str, cfg: Config) -> bool:
    """True when a missing env var belongs to something that is switched off."""
    trail = entry.split(" -> ")[0]
    if trail.endswith("web_search.api_key"):
        return cfg.tools.web_search.provider != "tavily"
    indexed = re.fullmatch(r"llm\.providers\[(\d+)\]\.api_key", trail)
    if indexed:
        provider = cfg.llm.providers[int(indexed.group(1))]
        return not (provider.enabled and provider.needs_key)
    return False


def _collect_warnings(cfg: Config, missing_env: list[str]) -> list[str]:
    warnings: list[str] = []

    required_missing = [entry for entry in missing_env if not _optional_env_ref(entry, cfg)]
    if required_missing:
        names = ", ".join(sorted({entry.split("$")[-1] for entry in required_missing}))
        warnings.append(
            f"env var(s) not set: {names} -> those values load as null "
            "(copy env.example to .env and fill what you need)"
        )
    if cfg.audio.wakeword.enabled and cfg.audio.frame_ms != 80:
        warnings.append(
            f"audio.frame_ms={cfg.audio.frame_ms}; openWakeWord wants 80 ms chunks, "
            "audio.py will buffer frames"
        )
    if not any(p.usable for p in cfg.llm.providers):
        warnings.append(
            "no usable llm provider: is `ollama serve` running, or a cloud key exported?"
        )
    local = [p for p in cfg.llm.providers if p.name == "ollama" and p.enabled]
    if cfg.llm.routing.mode in ("sequential", "local_only") and not local:
        warnings.append(
            f"routing.mode={cfg.llm.routing.mode} but the ollama provider is disabled"
        )

    voice = cfg.tts_model_path()
    if not voice.is_file():
        warnings.append(
            f"piper voice model missing: {voice} "
            "(download the voice into tts.voices_dir or set tts.model_path)"
        )
    for label, directory in (
        ("assistant.prompts_dir", cfg.assistant.prompts_dir),
        ("assistant.data_dir", cfg.assistant.data_dir),
        ("assistant.log_dir", cfg.assistant.log_dir),
        ("tools.notes.dir", cfg.tools.notes.dir),
    ):
        if not directory.is_dir():
            warnings.append(f"{label} {directory} does not exist yet; created on startup")

    seeds = cfg.memory.seed_facts
    if not seeds:
        warnings.append("memory.seed_facts is empty; Jarvis will start knowing nothing about you")
    else:
        for required in ("name", "tone", "assistant_name"):
            if required not in seeds:
                warnings.append(f"memory.seed_facts has no {required!r}; prompts will fall back to defaults")

    if "shell" in cfg.tools.enabled and not cfg.tools.shell.enabled:
        warnings.append("'shell' is listed in tools.enabled but tools.shell.enabled is false")
    if "shell" in cfg.tools.enabled and cfg.tools.shell.enabled and not cfg.tools.confirm_risky_actions:
        warnings.append(
            "shell commands run without confirmation (tools.confirm_risky_actions: false)"
        )
    if "web_search" in cfg.tools.enabled:
        search = cfg.tools.web_search
        if search.provider == "tavily" and not search.api_key:
            warnings.append("web_search provider is tavily but no TAVILY_API_KEY is set")
    if "home_assistant" in cfg.tools.enabled:
        hass = cfg.tools.home_assistant
        if not hass.url:
            warnings.append("home_assistant enabled but tools.home_assistant.url is null")
        elif not hass.token:
            warnings.append(
                f"home_assistant enabled but no token (HOME_ASSISTANT_TOKEN); "
                f"{hass.url} calls will fail"
            )
    return warnings


_CACHE: dict[Path, Config] = {}


def load_config(path: str | Path | None = None, *, reload: bool = False) -> Config:
    """Read, resolve and validate a config file. Cached per absolute path.

    Raises ConfigError with a readable message on any problem; nothing in the
    validation path raises a bare pydantic/yaml exception to the caller.
    """
    config_path = _config_path(path)
    if not reload and config_path in _CACHE:
        return _CACHE[config_path]

    if not config_path.is_file():
        raise ConfigError(
            f"config file not found: {config_path}\n"
            f"(pass --path, or set {CONFIG_ENV_VAR} to point at it)"
        )
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {config_path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML:\n{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")

    missing_env: list[str] = []
    data = _resolve_env_refs(raw, missing=missing_env)
    _resolve_paths(data, config_path.parent)

    try:
        cfg = Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, config_path)) from exc

    cfg = _absolutize_config_paths(cfg, config_path.parent)
    cfg = cfg.model_copy(
        update={
            "source_path": config_path,
            "warnings": tuple(_collect_warnings(cfg, missing_env)),
        }
    )
    _CACHE[config_path] = cfg
    return cfg


def _absolutize_config_paths(cfg: Config, base: Path) -> Config:
    """Anchor path defaults (keys missing from the yaml) to the config dir too."""
    for dotted in _PATH_KEYS:
        *parents, leaf = dotted.split(".")
        section: Any = cfg
        for part in parents:
            section = getattr(section, part)
        value = getattr(section, leaf)
        if isinstance(value, Path) and not value.is_absolute():
            setattr(section, leaf, _absolutize(str(value), base))
    return cfg


def get_config() -> Config:
    """Process-wide config: $JARVIS_CONFIG or ./config.yaml, loaded once."""
    return load_config()


def reset_cache() -> None:
    """Drop cached configs (tests, or after writing config.yaml)."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _secret_status(cfg: Config) -> list[tuple[str, bool]]:
    """Which optional secrets are present - names and booleans only, never values."""
    openai = cfg.provider("openai")
    anthropic = cfg.provider("anthropic")
    values = {
        "OPENAI_API_KEY": openai.api_key if openai else None,
        "ANTHROPIC_API_KEY": anthropic.api_key if anthropic else None,
        "HOME_ASSISTANT_TOKEN": cfg.tools.home_assistant.token,
        "TAVILY_API_KEY": cfg.tools.web_search.api_key,
    }
    return [(name, bool(value)) for name, value in values.items()]


def format_summary(cfg: Config) -> str:
    chain = " -> ".join(f"{p.name}:{p.model}" for p in cfg.routing_order()) or "none"
    wakeword = (
        "off"
        if not cfg.audio.wakeword.enabled
        else f"{cfg.audio.wakeword.model}@{cfg.audio.wakeword.threshold}"
    )
    seeds = len(cfg.memory.seed_facts)
    secrets = " ".join(f"{name}={'set' if ok else 'unset'}" for name, ok in _secret_status(cfg))
    return "\n".join(
        [
            f"config    {cfg.source_path}  (version {cfg.version})",
            f"assistant {cfg.assistant.name}  log_level={cfg.assistant.log_level}",
            f"paths     data={cfg.assistant.data_dir}",
            f"          logs={cfg.assistant.log_dir}  prompts={cfg.assistant.prompts_dir}",
            f"audio     {cfg.audio.sample_rate} Hz mono, {cfg.audio.frame_ms} ms frames, "
            f"wakeword={wakeword}, silence={cfg.audio.vad.silence_ms} ms",
            f"stt       {cfg.stt.engine} {cfg.stt.model} "
            f"(device={cfg.stt.device}, {cfg.stt.compute_type}, lang={cfg.stt.language or 'auto'})",
            f"llm       {chain}  [{cfg.llm.routing.mode}]",
            f"memory    sqlite={cfg.memory.sqlite_path}",
            f"          chroma={cfg.memory.chroma.path}  seeds={seeds}  decay={cfg.memory.decay_days}d",
            f"tts       {cfg.tts.engine} {cfg.tts.voice} @{cfg.tts.sample_rate} Hz",
            f"          -> {cfg.tts_model_path()}",
            f"tools     {', '.join(cfg.tools.enabled) or 'none'}",
            f"secrets   {secrets}",
        ]
    )


_SECRET_FIELD = re.compile(r"(api_key|token|secret|password)$", re.IGNORECASE)


def _redact(node: Any, field: str = "") -> Any:
    if isinstance(node, dict):
        return {key: _redact(value, str(key)) for key, value in node.items()}
    if isinstance(node, list):
        return [_redact(item, field) for item in node]
    if isinstance(node, str) and _SECRET_FIELD.search(field):
        return "<set>"
    return node


def redacted_config(cfg: Config) -> dict[str, Any]:
    """Config as JSON-safe data with every secret replaced by '<set>'."""
    return _redact(cfg.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="config.py", description="Validate Jarvis configuration and show what it resolved to."
    )
    parser.add_argument("-p", "--path", default=None, help="config file (default: ./config.yaml)")
    parser.add_argument("--check", action="store_true", help="validate and print a summary (default)")
    parser.add_argument("--json", action="store_true", help="print the resolved config as JSON")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.path, reload=True)
    except ConfigError as exc:
        print(f"config check FAILED\n{exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(redacted_config(cfg), indent=2))
    else:
        print(format_summary(cfg))
        if cfg.warnings:
            print(f"\n{len(cfg.warnings)} warning(s):")
            for warning in cfg.warnings:
                print(f"  ! {warning}")
        else:
            print("\nno warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
