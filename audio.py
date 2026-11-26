"""Audio front end: microphone capture -> wake word -> VAD -> speech-to-text.

The pipeline is built from small, injectable pieces so it can be tested without
a sound card:

    MicrophoneCapture   blocking iterator of 16-bit mono frames
    WakeWordDetector    openWakeWord scoring with threshold/patience/cooldown
    UtteranceCollector  energy VAD that cuts one spoken utterance out of frames
    Transcriber         faster-whisper wrapper
    VoiceListener       ties the four together and yields transcripts

Nothing heavy is imported at module import time: PortAudio, openWakeWord and
faster-whisper load on first use, so importing `audio` works on a headless box.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Iterator, Protocol

import numpy as np

from config import AudioConfig, SttConfig, VadConfig, WakeWordConfig

log = logging.getLogger(__name__)

# openWakeWord only ever consumes 1280-sample (80 ms at 16 kHz) chunks.
WAKE_CHUNK = 1280
PREROLL_MS = 240


class AudioError(RuntimeError):
    """Base class for audio problems."""


class AudioDeviceError(AudioError):
    """No usable microphone/speaker, or PortAudio cannot open one."""


def frame_size(sample_rate: int, frame_ms: int) -> int:
    """Samples in one frame; 16 kHz x 80 ms -> 1280."""
    return sample_rate * frame_ms // 1000


def rms(frame: np.ndarray) -> float:
    """Root-mean-square of an int16 frame, normalised to 0..1."""
    if frame.size == 0:
        return 0.0
    samples = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(samples))))


def to_float32(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 32768.0


@dataclass(frozen=True)
class Utterance:
    """One complete spoken segment, ready for transcription."""

    audio: np.ndarray
    speech_s: float
    sample_rate: int = 16_000

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


class UtteranceCollector:
    """Energy-based VAD: turn a frame stream into one utterance at a time.

    Frames before the first speech frame are kept as pre-roll so the opening
    syllable is not clipped. An utterance ends after `silence_ms` of quiet or
    at `max_utterance_s`, and is discarded if it held less than `min_speech_ms`
    of speech.
    """

    def __init__(
        self,
        cfg: VadConfig,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 80,
        preroll_ms: int = PREROLL_MS,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_s = frame_ms / 1000.0
        preroll_frames = max(0, round(preroll_ms / frame_ms))
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._frames: list[np.ndarray] = []
        self._speech_s = 0.0
        self._silence_s = 0.0
        self.reset()

    def reset(self) -> None:
        self._preroll.clear()
        self._frames = []
        self._speech_s = 0.0
        self._silence_s = 0.0

    @property
    def active(self) -> bool:
        return bool(self._frames)

    def add(self, frame: np.ndarray) -> Utterance | None:
        """Feed one frame; return an utterance when this frame completes one."""
        speech = rms(frame) >= self.cfg.energy_threshold

        if not self._frames:
            self._preroll.append(frame)
            if speech:
                self._frames = list(self._preroll)
                self._speech_s = self.frame_s
            return None

        self._frames.append(frame)
        if speech:
            self._speech_s += self.frame_s
            self._silence_s = 0.0
        else:
            self._silence_s += self.frame_s

        elapsed = len(self._frames) * self.frame_s
        quiet_end = self._silence_s + 1e-9 >= self.cfg.silence_ms / 1000.0
        too_long = elapsed + 1e-9 >= self.cfg.max_utterance_s
        if not (quiet_end or too_long):
            return None

        if self._speech_s + 1e-9 < self.cfg.min_speech_ms / 1000.0:
            self.reset()
            return None

        audio = np.concatenate(self._frames)
        speech_s = self._speech_s
        self.reset()
        return Utterance(audio=audio, speech_s=speech_s, sample_rate=self.sample_rate)


class WakeWordDetector:
    """openWakeWord with the threshold / patience / cooldown policy from config."""

    def __init__(
        self,
        cfg: WakeWordConfig,
        *,
        model_factory: Callable[[], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sample_rate: int = 16_000,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._model_factory = model_factory or (lambda: _load_wakeword_model(cfg.model))
        self._model: object | None = None
        self._buffer = np.zeros(0, dtype=np.int16)
        self._clock = clock
        self._above = 0
        self._last_hit = -math.inf

    def load(self) -> None:
        if self._model is None:
            log.info("loading wake word model %r", self.cfg.model)
            self._model = self._model_factory()

    def score(self, chunk: np.ndarray) -> float:
        self.load()
        scores = self._model.predict(chunk)  # type: ignore[attr-defined]
        return max(scores.values()) if scores else 0.0

    def feed(self, frame: np.ndarray) -> bool:
        """True exactly on the frame where the wake word should fire."""
        if not self.cfg.enabled:
            return False
        self._buffer = np.concatenate((self._buffer, frame.astype(np.int16, copy=False)))
        fired = False
        while len(self._buffer) >= WAKE_CHUNK:
            chunk = self._buffer[:WAKE_CHUNK]
            self._buffer = self._buffer[WAKE_CHUNK:]
            if self.score(chunk) >= self.cfg.threshold:
                self._above += 1
            else:
                self._above = 0
            now = self._clock()
            if self._above >= self.cfg.patience and now - self._last_hit >= self.cfg.cooldown_s:
                self._last_hit = now
                self._above = 0
                fired = True
        return fired


class Transcriber:
    """faster-whisper wrapper; the model loads once, on first use."""

    def __init__(self, cfg: SttConfig, *, model_factory: Callable[[], object] | None = None) -> None:
        self.cfg = cfg
        self._factory = model_factory or self._load_model
        self._model: object | None = None

    def _load_model(self) -> object:
        from faster_whisper import WhisperModel

        log.info("loading whisper model %r (%s/%s)", self.cfg.model, self.cfg.device, self.cfg.compute_type)
        return WhisperModel(
            self.cfg.model,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
            cpu_threads=self.cfg.cpu_threads,
            download_root=str(self.cfg.download_root),
        )

    def load(self) -> None:
        if self._model is None:
            self._model = self._factory()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe int16 PCM; returns '' for silence or an empty result."""
        if audio.size == 0:
            return ""
        self.load()
        segments, _info = self._model.transcribe(  # type: ignore[attr-defined]
            to_float32(audio),
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


class FrameSource(Protocol):
    def frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class MicrophoneCapture:
    """sounddevice input stream yielding int16 mono frames of a fixed size."""

    def __init__(
        self,
        cfg: AudioConfig,
        *,
        sd_module: object | None = None,
        stream_factory: Callable[..., object] | None = None,
    ) -> None:
        self.cfg = cfg
        self._sd = sd_module
        self._stream_factory = stream_factory
        self._blocksize = frame_size(cfg.sample_rate, cfg.frame_ms)

    def _sounddevice(self) -> object:
        if self._sd is None:
            try:
                import sounddevice
            except (ImportError, OSError) as exc:  # PortAudio missing / no server
                raise AudioDeviceError(f"cannot initialise PortAudio: {exc}") from exc
            self._sd = sounddevice
        return self._sd

    def resolve_device(self) -> int | None:
        """Config value -> sounddevice index. Strings match device names."""
        wanted = self.cfg.input_device
        if wanted is None or isinstance(wanted, int):
            return wanted
        sd = self._sounddevice()
        try:
            devices = sd.query_devices()
        except Exception as exc:  # PortAudioError and friends
            raise AudioDeviceError(f"cannot list audio devices: {exc}") from exc
        matches = [
            index
            for index, info in enumerate(devices)
            if info.get("max_input_channels", 0) > 0 and wanted.lower() in str(info["name"]).lower()
        ]
        if not matches:
            raise AudioDeviceError(f"no input device matches {wanted!r}")
        return matches[0]

    def frames(self) -> Iterator[np.ndarray]:
        sd = self._sounddevice()
        device = self.resolve_device()
        factory = self._stream_factory or sd.InputStream
        try:
            stream = factory(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="int16",
                blocksize=self._blocksize,
                device=device,
            )
        except Exception as exc:
            raise AudioDeviceError(f"cannot open microphone: {exc}") from exc
        log.info("microphone open: %s Hz, %d-sample frames", self.cfg.sample_rate, self._blocksize)
        with stream:
            while True:
                data, overflowed = stream.read(self._blocksize)
                if overflowed:
                    log.debug("microphone overflow; %d frames dropped", overflowed)
                frame = np.asarray(data, dtype=np.int16).reshape(-1)
                if frame.size:
                    yield frame

    def close(self) -> None:
        """Nothing to do: the stream closes when `frames()` is exhausted."""


class VoiceListener:
    """Wake word -> utterance collection -> transcription, as a text stream."""

    def __init__(
        self,
        capture: FrameSource,
        collector: UtteranceCollector,
        transcriber: Transcriber,
        wakeword: WakeWordDetector | None = None,
    ) -> None:
        self.capture = capture
        self.collector = collector
        self.transcriber = transcriber
        self.wakeword = wakeword if wakeword is not None and wakeword.cfg.enabled else None

    def utterances(self) -> Iterator[str]:
        armed = self.wakeword is None
        for frame in self.capture.frames():
            if not armed:
                if self.wakeword and self.wakeword.feed(frame):
                    log.info("wake word detected")
                    armed = True
                    self.collector.reset()
                continue
            utterance = self.collector.add(frame)
            if utterance is None:
                continue
            armed = self.wakeword is None
            if utterance.duration_s <= 0:
                continue
            text = self.transcriber.transcribe(utterance.audio)
            if text:
                log.info("heard %r (%.1fs)", text, utterance.duration_s)
                yield text

    async def listen(self) -> AsyncIterator[str]:
        """Async wrapper: one thread pumps blocking frames into a queue."""
        import asyncio

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        iterator = self.utterances()

        def pump() -> None:
            try:
                for text in iterator:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:  # surfaced in the async loop below
                loop.call_soon_threadsafe(queue.put_nowait, exc)  # type: ignore[arg-type]
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(pump))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            task.cancel()
            self.capture.close()


def build_listener(cfg: AudioConfig, stt: SttConfig) -> VoiceListener:
    """Wire the real microphone, wake word and Whisper from config."""
    capture = MicrophoneCapture(cfg)
    collector = UtteranceCollector(cfg.vad, sample_rate=cfg.sample_rate, frame_ms=cfg.frame_ms)
    wakeword = WakeWordDetector(cfg.wakeword, sample_rate=cfg.sample_rate) if cfg.wakeword.enabled else None
    return VoiceListener(capture, collector, Transcriber(stt), wakeword)


def _wakeword_model_dir() -> Path:
    import openwakeword

    return Path(openwakeword.__file__).resolve().parent / "resources" / "models"


def _download_wakeword_model(name: str) -> None:
    from openwakeword.utils import download_models

    download_models(model_names=[name])


def ensure_wakeword_model(name: str, *, download: bool = True) -> Path:
    """Path to openWakeWord's bundled <name>_v*.onnx, fetching it once when absent.

    A bare name such as 'alexa' or 'hey_jarvis' used to be handed to
    `Model`, which looks for the file next to the installed package and fails
    with "File doesn't exist" when it was never downloaded - taking the whole
    voice loop down with it.
    """
    directory = _wakeword_model_dir()
    found = sorted(directory.glob(f"{name}_v*.onnx"))
    if found:
        return found[-1]
    if download:
        try:
            log.info("downloading wake word model %r", name)
            _download_wakeword_model(name)
        except Exception as exc:  # network down, unknown name, no write access
            raise AudioError(
                f"could not download wake word model {name!r}: {exc}"
            ) from exc
        found = sorted(directory.glob(f"{name}_v*.onnx"))
    if not found:
        raise AudioError(
            f"wake word model {name!r} is not available; run 'python doctor.py', "
            "or set audio.wakeword.model to a .onnx file"
        )
    return found[-1]


def _load_wakeword_model(model: str | None = None) -> object:
    name = model
    if name is None:
        raise AudioError("wake word model name required")
    explicit = name.endswith(".onnx") or "/" in name or "\\" in name
    if explicit and not Path(name).exists():
        raise AudioError(f"wake word model file {name} does not exist")

    from openwakeword.model import Model

    if explicit:
        return Model(wakeword_models=[str(Path(name))])
    return Model(wakeword_models=[str(ensure_wakeword_model(name))], inference_framework="onnx")