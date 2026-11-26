"""Preflight checks for Jarvis: downloads the Piper voice, inspects audio.

    python doctor.py                 # check everything; download a missing voice
    python doctor.py --no-download   # checks only
    python doctor.py --json          # machine-readable report

Exit codes: 0 everything ready, 1 warnings remain, 2 invalid configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import httpx

from config import Config, ConfigError, load_config

VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
REQUIRED_MODULES = ("numpy", "yaml", "pydantic", "httpx", "faster_whisper", "openwakeword", "chromadb")
Status = Literal["ok", "warn", "fail"]


class DoctorError(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    status: Status
    detail: str


def voice_urls(voice: str) -> tuple[str, str]:
    """`en_US-lessac-medium` -> the two Hugging Face URLs for that voice."""
    parts = voice.split("-")
    if len(parts) != 3 or "_" not in parts[0]:
        raise DoctorError(f"cannot derive a download URL from voice {voice!r}; use locale-name-quality")
    locale, name, quality = parts
    language = locale.split("_", 1)[0]
    stem = f"{language}/{locale}/{name}/{quality}/{voice}"
    return f"{VOICE_BASE}/{stem}.onnx", f"{VOICE_BASE}/{stem}.onnx.json"


def download_voice(
    cfg: Config,
    *,
    client: httpx.Client | None = None,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Fetch the configured voice (onnx + json) into voices_dir if missing."""
    if cfg.tts.model_path is not None:
        raise DoctorError("tts.model_path is set; download that voice manually")
    onnx_url, json_url = voice_urls(cfg.tts.voice)
    directory = Path(cfg.tts.voices_dir)
    directory.mkdir(parents=True, exist_ok=True)
    targets = [
        (onnx_url, directory / f"{cfg.tts.voice}.onnx"),
        (json_url, directory / f"{cfg.tts.voice}.onnx.json"),
    ]
    owned = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=10.0))
    written: list[Path] = []
    try:
        for url, path in targets:
            if path.exists() and path.stat().st_size > 0:
                continue
            temporary = path.with_name(path.name + ".part")
            with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise DoctorError(f"download failed ({response.status_code}) for {url}")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise DoctorError(f"download produced an empty file for {url}")
            temporary.replace(path)
            written.append(path)
            log(f"downloaded {path.name} ({path.stat().st_size:,} bytes)")
    finally:
        if owned:
            client.close()
    return written


def audio_devices(sd_module: object | None = None) -> tuple[list[str], list[str]]:
    if sd_module is None:
        try:
            import sounddevice as sd_module
        except Exception as exc:  # PortAudioError, ImportError, OSError all mean unusable
            raise DoctorError(f"PortAudio unavailable: {exc}") from exc
    try:
        devices = sd_module.query_devices()  # type: ignore[attr-defined]
    except Exception as exc:
        raise DoctorError(f"cannot list audio devices: {exc}") from exc
    inputs = [str(d.get("name", "?")) for d in devices if d.get("max_input_channels", 0) > 0]
    outputs = [str(d.get("name", "?")) for d in devices if d.get("max_output_channels", 0) > 0]
    return inputs, outputs


def check_python() -> Check:
    if sys.version_info[:2] >= (3, 11):
        return Check("python", "ok", f"{sys.version_info.major}.{sys.version_info.minor}")
    return Check("python", "fail", f"{sys.version_info.major}.{sys.version_info.minor}; needs 3.11+")


def check_modules() -> Check:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        return Check("dependencies", "warn", "missing: " + ", ".join(missing) + " (pip install -r requirements.txt)")
    return Check("dependencies", "ok", f"{len(REQUIRED_MODULES)} modules present")


def check_piper_binary(which: Callable[[str], str | None] = shutil.which) -> Check:
    binary = which("piper")
    return Check("piper", "ok" if binary else "warn", binary or "piper binary not found on PATH")


def check_voice(cfg: Config, *, download: bool, log: Callable[[str], None] = print) -> Check:
    model = Path(cfg.tts.model_path) if cfg.tts.model_path is not None else Path(cfg.tts.voices_dir) / f"{cfg.tts.voice}.onnx"
    if model.exists() and model.stat().st_size > 0:
        return Check("voice", "ok", str(model))
    if not download:
        return Check("voice", "warn", f"missing {model}; rerun without --no-download")
    try:
        download_voice(cfg, log=log)
    except (DoctorError, httpx.HTTPError) as exc:
        return Check("voice", "warn", f"could not download {cfg.tts.voice}: {exc}")
    if model.exists() and model.stat().st_size > 0:
        return Check("voice", "ok", str(model))
    return Check("voice", "warn", f"still missing {model}")


def check_audio(sd_module: object | None = None) -> Check:
    try:
        inputs, outputs = audio_devices(sd_module)
    except DoctorError as exc:
        return Check("audio", "warn", f"{exc} (voice mode needs a sound card)")
    detail = f"{len(inputs)} input / {len(outputs)} output devices"
    if not inputs:
        return Check("audio", "warn", detail + "; no microphone found")
    return Check("audio", "ok", detail + f"; mic: {inputs[0]}")


def check_ollama(cfg: Config, *, client: httpx.Client | None = None) -> Check:
    provider = cfg.provider("ollama")
    if provider is None or not provider.enabled:
        return Check("ollama", "ok", "not configured")
    base = (provider.base_url or "http://localhost:11434").rstrip("/")
    owned = client is None
    client = client or httpx.Client(timeout=3.0)
    try:
        response = client.get(f"{base}/api/tags")
        if response.status_code >= 400:
            return Check("ollama", "warn", f"HTTP {response.status_code} from {base}")
        names = [item.get("name", "") for item in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as exc:
        return Check("ollama", "warn", f"{base} unreachable: {exc}")
    finally:
        if owned:
            client.close()
    if provider.model in names:
        return Check("ollama", "ok", f"{provider.model} available")
    if names:
        return Check("ollama", "warn", f"{provider.model} not pulled; have: {', '.join(names[:5])}")
    return Check("ollama", "warn", f"no models installed; run: ollama pull {provider.model}")


def check_keys(cfg: Config) -> Check:
    unset = []
    for provider in cfg.llm.providers:
        if provider.name == "ollama":
            continue
        if provider.enabled and not provider.api_key:
            unset.append(f"{provider.name} ({provider.name.upper()}_API_KEY)")
    if cfg.tools.home_assistant.token is None:
        unset.append("home_assistant (HOME_ASSISTANT_TOKEN)")
    if unset:
        return Check("keys", "warn", "unset: " + ", ".join(unset))
    return Check("keys", "ok", "cloud/HA secrets present")


def check_wakeword(cfg: Config, *, download: bool) -> Check:
    if not cfg.audio.wakeword.enabled:
        return Check("wakeword", "ok", "disabled")
    name = cfg.audio.wakeword.model
    if name.endswith(".onnx") or "/" in name or "\\" in name:
        path = Path(name)
        return Check("wakeword", "ok" if path.exists() else "warn", str(path))
    try:
        from audio import ensure_wakeword_model

        path = ensure_wakeword_model(name, download=download)
    except ImportError:
        return Check("wakeword", "warn", "dependencies missing (pip install -r requirements.txt)")
    except Exception as exc:  # noqa: BLE001 - a check reports, it never raises
        return Check("wakeword", "warn", f"could not download {name}: {exc}")
    return Check("wakeword", "ok", str(path))


def run_checks(cfg: Config, *, download: bool, sd_module: object | None = None) -> list[Check]:
    return [
        check_python(),
        check_modules(),
        check_piper_binary(),
        check_voice(cfg, download=download),
        check_wakeword(cfg, download=download),
        check_audio(sd_module),
        check_ollama(cfg),
        check_keys(cfg),
    ]


def summarize(checks: Iterable[Check]) -> int:
    checks = list(checks)
    if any(check.status == "fail" for check in checks):
        return 1
    return 1 if any(check.status == "warn" for check in checks) else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Jarvis's environment and fetch a missing voice")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--no-download", action="store_true", help="checks only; do not fetch the Piper voice")
    parser.add_argument("--json", action="store_true", help="print JSON instead of lines")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    checks = run_checks(cfg, download=not args.no_download)
    if args.json:
        print(json.dumps({"checks": [asdict(check) for check in checks]}, indent=2))
    else:
        for check in checks:
            print(f"[{check.status:4}] {check.name:12} {check.detail}")
        warnings = sum(1 for check in checks if check.status != "ok")
        print("ready" if warnings == 0 else f"{warnings} item(s) need attention")
    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
