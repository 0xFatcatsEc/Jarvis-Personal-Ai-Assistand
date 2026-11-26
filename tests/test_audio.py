"""Tests for audio.py: framing, VAD, wake word policy, transcription wiring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import islice

import numpy as np
import pytest

import audio
from audio import (
    MicrophoneCapture,
    Transcriber,
    UtteranceCollector,
    VoiceListener,
    WakeWordDetector,
    frame_size,
    rms,
)
from config import AudioConfig, SttConfig, VadConfig, WakeWordConfig

FRAME_MS = 80
FRAME_SAMPLES = 1280
SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.int16)
SPEECH = np.full(FRAME_SAMPLES, 6000, dtype=np.int16)


def vad_cfg(**overrides) -> VadConfig:
    values = dict(energy_threshold=0.012, silence_ms=800, min_speech_ms=300, max_utterance_s=15.0)
    values.update(overrides)
    return VadConfig(**values)


def collector(**overrides) -> UtteranceCollector:
    return UtteranceCollector(vad_cfg(**overrides), frame_ms=FRAME_MS)


def test_frame_size_matches_openwakeword_chunk():
    assert frame_size(16_000, 80) == 1280
    assert frame_size(16_000, 20) == 320


def test_rms_normalises_int16():
    assert rms(SILENCE) == 0.0
    assert rms(np.full(100, 32767, dtype=np.int16)) == pytest.approx(1.0, abs=1e-3)
    assert rms(np.array([], dtype=np.int16)) == 0.0


def test_collector_returns_utterance_after_trailing_silence():
    vad = collector()
    result = None
    for frame in [*[SPEECH] * 5, *[SILENCE] * 10]:
        result = vad.add(frame) or result
    assert result is not None
    assert result.speech_s == pytest.approx(0.4)
    # 5 speech + 10 silence frames, with the 3-frame pre-roll already inside.
    assert result.duration_s == pytest.approx(15 * 0.08)
    assert result.audio.dtype == np.int16


def test_collector_keeps_preroll_before_onset():
    vad = collector()
    for _ in range(3):
        assert vad.add(SILENCE) is None
    result = None
    for frame in [*[SPEECH] * 5, *[SILENCE] * 10]:
        result = vad.add(frame) or result
    # 2 pre-roll + 5 speech + 10 silence: the 3-frame window ends on the onset
    # frame, so the oldest pre-roll frame has slid out.
    assert result is not None
    assert len(result.audio) == 17 * FRAME_SAMPLES


def test_collector_discards_short_blips():
    vad = collector()
    result = None
    for frame in [*[SPEECH] * 2, *[SILENCE] * 30]:
        result = vad.add(frame) or result
    assert result is None
    assert not vad.active


def test_collector_forces_end_at_max_utterance():
    vad = collector(max_utterance_s=0.5)
    result = None
    for _ in range(20):
        result = vad.add(SPEECH) or result
        if result:
            break
    assert result is not None
    assert result.duration_s >= 0.5
    assert not vad.active


def test_collector_discards_max_length_utterance_without_speech():
    vad = collector(max_utterance_s=0.5, energy_threshold=0.5)
    for _ in range(10):
        assert vad.add(SILENCE) is None


class FakeWakeModel:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    def predict(self, chunk):
        self.calls.append(np.asarray(chunk).shape)
        return {"hey_jarvis": next(self.scores, 0.0)}


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def test_wakeword_needs_patience_consecutive_frames():
    model = FakeWakeModel([0.9, 0.9, 0.9])
    detector = WakeWordDetector(WakeWordConfig(threshold=0.55, patience=2, cooldown_s=0.0), model_factory=lambda: model)
    assert detector.feed(SPEECH) is False
    assert detector.feed(SPEECH) is True
    assert detector.feed(SPEECH) is False


def test_wakeword_breaks_patience_streak():
    model = FakeWakeModel([0.9, 0.1, 0.9, 0.9])
    detector = WakeWordDetector(WakeWordConfig(threshold=0.55, patience=2, cooldown_s=0.0), model_factory=lambda: model)
    assert detector.feed(SPEECH) is False
    assert detector.feed(SPEECH) is False
    assert detector.feed(SPEECH) is False
    assert detector.feed(SPEECH) is True


def test_wakeword_respects_cooldown():
    clock = FakeClock()
    model = FakeWakeModel([0.9, 0.9, 0.9, 0.9])
    detector = WakeWordDetector(
        WakeWordConfig(threshold=0.55, patience=1, cooldown_s=2.0),
        model_factory=lambda: model,
        clock=clock,
    )
    assert detector.feed(SPEECH) is True
    clock.now = 1.0
    assert detector.feed(SPEECH) is False
    clock.now = 2.1
    assert detector.feed(SPEECH) is True


def test_wakeword_buffers_small_frames_into_1280_sample_chunks():
    model = FakeWakeModel([0.0, 0.9, 0.9])
    detector = WakeWordDetector(WakeWordConfig(threshold=0.5, patience=1, cooldown_s=0.0), model_factory=lambda: model)
    quarter = np.zeros(320, dtype=np.int16)
    fired = [detector.feed(quarter) for _ in range(8)]
    assert model.calls == [(1280,), (1280,)]
    assert fired.count(True) == 1


def test_wakeword_disabled_never_fires():
    detector = WakeWordDetector(WakeWordConfig(enabled=False), model_factory=lambda: FakeWakeModel([1.0] * 4))
    assert detector.feed(SPEECH) is False


@dataclass
class FakeSegment:
    text: str


class FakeWhisperModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([FakeSegment(text=s) for s in self.segments]), object()


def test_transcriber_joins_segments_and_passes_settings():
    model = FakeWhisperModel([" hello ", "", "world "])
    cfg = SttConfig(model="base.en", language="en", beam_size=3, vad_filter=False)
    transcriber = Transcriber(cfg, model_factory=lambda: model)
    assert transcriber.transcribe(SPEECH) == "hello world"
    audio_arg, kwargs = model.calls[0]
    assert audio_arg.dtype == np.float32
    assert kwargs["beam_size"] == 3 and kwargs["vad_filter"] is False
    assert kwargs["language"] == "en"


def test_transcriber_empty_audio_returns_empty_without_loading():
    cfg = SttConfig()
    transcriber = Transcriber(cfg, model_factory=lambda: pytest.fail("must not load"))
    assert transcriber.transcribe(np.array([], dtype=np.int16)) == ""


class FakeCapture:
    def __init__(self, frames):
        self._frames = frames
        self.closed = False

    def frames(self):
        yield from self._frames

    def close(self):
        self.closed = True


class FakeTranscriber:
    def __init__(self, text="hello there"):
        self.text = text
        self.received = []

    def transcribe(self, audio):
        self.received.append(audio)
        return self.text


def test_listener_transcribes_one_utterance():
    capture = FakeCapture([*[SILENCE] * 2, *[SPEECH] * 5, *[SILENCE] * 10])
    transcriber = FakeTranscriber("turn on the lights")
    listener = VoiceListener(capture, collector(), transcriber)
    assert list(listener.utterances()) == ["turn on the lights"]
    assert len(transcriber.received) == 1


def test_listener_waits_for_wake_word():
    class OnceWake:
        cfg = WakeWordConfig()

        def __init__(self):
            self.calls = 0

        def feed(self, frame):
            self.calls += 1
            return self.calls == 3

    capture = FakeCapture([*[SILENCE] * 3, *[SPEECH] * 5, *[SILENCE] * 10])
    listener = VoiceListener(capture, collector(), FakeTranscriber("yo"), OnceWake())
    assert list(listener.utterances()) == ["yo"]


def test_listener_async_wrapper_yields_and_closes():
    capture = FakeCapture([*[SPEECH] * 5, *[SILENCE] * 10])
    listener = VoiceListener(capture, collector(), FakeTranscriber("async hi"))

    async def collect():
        return [text async for text in listener.listen()]

    assert asyncio.run(collect()) == ["async hi"]
    assert capture.closed is True


class FakeStream:
    def __init__(self, frames):
        self._frames = iter(frames)
        self._open = False

    def __enter__(self):
        self._open = True
        return self

    def __exit__(self, *exc):
        self._open = False

    def read(self, size):
        return np.zeros((size, 1), dtype=np.int16), False


class FakeSd:
    def __init__(self):
        self.devices = [
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "Blue Yeti Microphone", "max_input_channels": 2},
        ]

    def query_devices(self):
        return self.devices


def test_capture_resolves_device_by_name():
    cfg = AudioConfig(input_device="yeti")
    capture = MicrophoneCapture(cfg, sd_module=FakeSd())
    assert capture.resolve_device() == 1


def test_capture_rejects_unknown_device_name():
    cfg = AudioConfig(input_device="nope")
    capture = MicrophoneCapture(cfg, sd_module=FakeSd())
    with pytest.raises(audio.AudioDeviceError, match="no input device"):
        capture.resolve_device()


def test_capture_yields_mono_int16_frames():
    calls = {}

    def factory(**kwargs):
        calls.update(kwargs)
        return FakeStream([np.zeros((1280, 1), dtype=np.int16)])

    cfg = AudioConfig(input_device=2, sample_rate=16_000, frame_ms=80)
    capture = MicrophoneCapture(cfg, sd_module=FakeSd(), stream_factory=factory)
    frames = list(islice(capture.frames(), 2))
    assert calls["blocksize"] == 1280 and calls["device"] == 2
    assert calls["samplerate"] == 16_000 and calls["channels"] == 1
    assert all(frame.dtype == np.int16 and frame.shape == (1280,) for frame in frames)


# --------------------------------------------------------------------------- #
# wake word model resolution
# --------------------------------------------------------------------------- #
def test_ensure_wakeword_model_uses_a_bundled_file(tmp_path, monkeypatch):
    (tmp_path / "hey_jarvis_v0.1.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(audio, "_wakeword_model_dir", lambda: tmp_path)
    monkeypatch.setattr(
        audio, "_download_wakeword_model", lambda name: pytest.fail("must not download")
    )

    assert audio.ensure_wakeword_model("hey_jarvis") == tmp_path / "hey_jarvis_v0.1.onnx"


def test_ensure_wakeword_model_downloads_a_missing_file(tmp_path, monkeypatch):
    calls = []

    def fake_download(name):
        calls.append(name)
        (tmp_path / f"{name}_v0.1.onnx").write_bytes(b"onnx")

    monkeypatch.setattr(audio, "_wakeword_model_dir", lambda: tmp_path)
    monkeypatch.setattr(audio, "_download_wakeword_model", fake_download)

    path = audio.ensure_wakeword_model("alexa")
    assert path.name == "alexa_v0.1.onnx" and calls == ["alexa"]


def test_ensure_wakeword_model_explains_an_unknown_name(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "_wakeword_model_dir", lambda: tmp_path)
    monkeypatch.setattr(audio, "_download_wakeword_model", lambda name: None)

    with pytest.raises(audio.AudioError, match="not available"):
        audio.ensure_wakeword_model("nope")


def test_load_wakeword_model_rejects_a_missing_file(tmp_path):
    with pytest.raises(audio.AudioError, match="does not exist"):
        audio._load_wakeword_model(str(tmp_path / "gone.onnx"))
