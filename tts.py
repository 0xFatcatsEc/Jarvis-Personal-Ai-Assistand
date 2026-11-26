"""Text to speech with Piper, plus sentence chunking and playback.

Synthesis shells out to the `piper` binary and reads the WAV it writes to a
temporary file (piper-tts 1.3 cannot stream WAV to stdout), so no Piper Python
API churn. Playback uses sounddevice, which is imported lazily: on a headless
box `is_available()` is False and `main` falls back to printing text.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from config import TtsConfig

log = logging.getLogger(__name__)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MAX_SYNTH_CHARS = 2_000


class TTSError(RuntimeError):
    """Piper is missing, the voice is missing, or synthesis failed."""


@dataclass(frozen=True)
class Speech:
    samples: np.ndarray  # int16, mono
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def chunk_text(text: str, *, max_chars: int = 240) -> list[str]:
    """Split into speakable sentences, packing short ones up to max_chars."""
    clean = " ".join(text.split())
    if not clean:
        return []
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(clean) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _run_piper(command: Sequence[str], text: str, timeout_s: float) -> bytes:
    try:
        result = subprocess.run(
            list(command),
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise TTSError(f"piper is not installed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TTSError(f"piper timed out after {timeout_s}s") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise TTSError(f"piper failed (exit {result.returncode}): {detail[-1] if detail else 'no output'}")
    if not result.stdout and command[-1] == "-":
        raise TTSError("piper produced no audio")
    return result.stdout


class PiperEngine:
    def __init__(
        self,
        cfg: TtsConfig,
        *,
        which: Callable[[str], str | None] = shutil.which,
        runner: Callable[[Sequence[str], str, float], bytes] | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.cfg = cfg
        self._which = which
        self._runner = runner or _run_piper
        self.timeout_s = timeout_s

    def model_path(self) -> Path:
        if self.cfg.model_path is not None:
            return Path(self.cfg.model_path)
        return Path(self.cfg.voices_dir) / f"{self.cfg.voice}.onnx"

    def is_available(self) -> bool:
        return self._which("piper") is not None and self.model_path().exists()

    def command(self, *, speed: float | None = None, output_file: str = "-") -> list[str]:
        binary = self._which("piper")
        if binary is None:
            raise TTSError("piper binary not found on PATH; install piper-tts")
        model = self.model_path()
        if not model.exists():
            raise TTSError(
                f"voice model {model} is missing; download {self.cfg.voice}.onnx "
                f"and its .json into {self.cfg.voices_dir}"
            )
        rate = self.cfg.speed if speed is None else speed
        length_scale = 1.0 / max(0.5, min(2.0, rate))
        return [
            binary,
            "--model",
            str(model),
            "--length-scale",
            f"{length_scale:.4f}",
            "--output-file",
            output_file,
        ]

    def synthesize(self, text: str, *, speed: float | None = None) -> Speech:
        clean = " ".join(text.split())
        if not clean:
            raise TTSError("nothing to speak")
        if len(clean) > MAX_SYNTH_CHARS:
            raise TTSError(f"text is longer than {MAX_SYNTH_CHARS} characters")
        # piper-tts 1.3 crashes writing WAV to stdout ('Wave_write' has no
        # attribute '_file'), so it writes a file and Jarvis reads that.
        with tempfile.TemporaryDirectory(prefix="jarvis-tts-") as directory:
            target = Path(directory) / "speech.wav"
            wav_bytes = self._runner(
                self.command(speed=speed, output_file=str(target)), clean, self.timeout_s
            )
            if target.exists() and target.stat().st_size:
                wav_bytes = target.read_bytes()
        return self._decode(wav_bytes)

    def _decode(self, wav_bytes: bytes) -> Speech:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError) as exc:
            raise TTSError(f"piper returned invalid WAV: {exc}") from exc
        if width != 2:
            raise TTSError(f"expected 16-bit audio, got {width * 8}-bit")
        samples = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
        volume = self.cfg.volume
        if volume != 1.0:
            samples = np.clip(samples.astype(np.float32) * volume, -32768, 32767).astype(np.int16)
        return Speech(samples=samples.copy(), sample_rate=rate)


class Speaker:
    """Plays `Speech` through sounddevice (or an injected playback function)."""

    def __init__(
        self,
        cfg: TtsConfig,
        *,
        engine: PiperEngine | None = None,
        sd_module: object | None = None,
        play: Callable[..., None] | None = None,
        wait: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine or PiperEngine(cfg)
        self._sd = sd_module
        self._play = play
        self._wait = wait

    def _sounddevice(self) -> object:
        if self._sd is None:
            try:
                import sounddevice
            except (ImportError, OSError) as exc:
                raise TTSError(f"cannot initialise audio output: {exc}") from exc
            self._sd = sounddevice
        return self._sd

    @property
    def available(self) -> bool:
        try:
            return self.engine.is_available()
        except TTSError:
            return False

    def speak(self, text: str) -> None:
        speech = self.engine.synthesize(text)
        play = self._play
        wait = self._wait
        if play is None:
            sd = self._sounddevice()
            play = sd.play
            wait = wait or sd.wait
        play(speech.samples, samplerate=speech.sample_rate, device=self.cfg.device_index)
        if wait is not None:
            wait()
        log.debug("spoke %d chars (%.2fs audio)", len(text), speech.duration_s)

    def speak_chunks(self, text: str, *, max_chars: int = 240) -> None:
        for chunk in chunk_text(text, max_chars=max_chars):
            self.speak(chunk)
