"""Tests for tts.py: chunking, Piper invocation, WAV decoding and playback."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from config import TtsConfig
from tts import PiperEngine, Speaker, Speech, TTSError, chunk_text


def wav_bytes(samples: np.ndarray, rate: int = 22_050, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


def tone(n: int = 100, value: int = 1000) -> np.ndarray:
    return np.full(n, value, dtype=np.int16)


@pytest.fixture()
def voice(tmp_path):
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"fake model")
    return model


def engine(tmp_path, voice, runner, **cfg) -> PiperEngine:
    config = TtsConfig(voices_dir=tmp_path, model_path=voice, **cfg)
    return PiperEngine(config, which=lambda name: "/usr/bin/piper", runner=runner)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def test_chunk_text_packs_sentences():
    assert chunk_text("Hello there. How are you?", max_chars=240) == ["Hello there. How are you?"]


def test_chunk_text_splits_when_over_budget():
    chunks = chunk_text("Hello there. How are you? Fine!", max_chars=15)
    assert chunks == ["Hello there.", "How are you?", "Fine!"]


def test_chunk_text_hard_splits_long_sentence():
    text = " ".join(["word"] * 20)
    chunks = chunk_text(text, max_chars=20)
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert " ".join(chunks) == text


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_chunk_text_empty(text):
    assert chunk_text(text) == []


def test_chunk_text_normalises_whitespace():
    assert chunk_text("  Hello \n  world.  ") == ["Hello world."]


# --------------------------------------------------------------------------- #
# synthesis
# --------------------------------------------------------------------------- #
def test_synthesize_decodes_wav_and_uses_speed(voice, tmp_path):
    calls = {}

    def runner(command, text, timeout):
        calls["command"] = command
        calls["text"] = text
        return wav_bytes(tone(220))

    synth = engine(tmp_path, voice, runner, speed=1.2)
    speech = synth.synthesize("hi there")
    assert isinstance(speech, Speech)
    assert speech.sample_rate == 22_050
    assert speech.samples.dtype == np.int16
    assert len(speech.samples) == 220
    assert speech.duration_s == pytest.approx(220 / 22_050)
    assert calls["text"] == "hi there"
    assert "--length-scale" in calls["command"]
    assert calls["command"][calls["command"].index("--length-scale") + 1] == "0.8333"
    assert calls["command"][-1] == "-"


def test_synthesize_applies_volume(voice, tmp_path):
    synth = engine(tmp_path, voice, lambda c, t, s: wav_bytes(tone(50, 1000)), volume=0.5)
    speech = synth.synthesize("quiet")
    assert set(np.unique(speech.samples)) == {500}


def test_synthesize_mixes_stereo(voice, tmp_path):
    stereo = np.stack([tone(10, 100), tone(10, 300)], axis=1).reshape(-1)
    synth = engine(tmp_path, voice, lambda c, t, s: wav_bytes(stereo, channels=2))
    assert np.all(synth.synthesize("x").samples == 200)


def test_missing_model_is_a_clear_error(tmp_path):
    synth = PiperEngine(
        TtsConfig(voices_dir=tmp_path, model_path=tmp_path / "nope.onnx"),
        which=lambda name: "/usr/bin/piper",
        runner=lambda c, t, s: b"",
    )
    with pytest.raises(TTSError, match="missing"):
        synth.synthesize("hello")


def test_missing_binary_is_a_clear_error(voice, tmp_path):
    synth = PiperEngine(TtsConfig(model_path=voice), which=lambda name: None)
    with pytest.raises(TTSError, match="not found"):
        synth.synthesize("hello")


def test_invalid_wav_is_reported(voice, tmp_path):
    synth = engine(tmp_path, voice, lambda c, t, s: b"not a wav")
    with pytest.raises(TTSError, match="invalid WAV"):
        synth.synthesize("hello")


def test_empty_text_refused(voice, tmp_path):
    synth = engine(tmp_path, voice, lambda c, t, s: wav_bytes(tone()))
    with pytest.raises(TTSError, match="nothing to speak"):
        synth.synthesize("   ")


def test_overlong_text_refused(voice, tmp_path):
    synth = engine(tmp_path, voice, lambda c, t, s: wav_bytes(tone()))
    with pytest.raises(TTSError, match="longer"):
        synth.synthesize("x" * 2001)


def test_is_available_checks_binary_and_model(tmp_path, voice):
    assert engine(tmp_path, voice, lambda c, t, s: b"").is_available() is True
    absent = PiperEngine(TtsConfig(voices_dir=tmp_path, model_path=tmp_path / "x.onnx"), which=lambda n: "/bin/piper")
    assert absent.is_available() is False


# --------------------------------------------------------------------------- #
# playback
# --------------------------------------------------------------------------- #
class FakeEngine:
    available = True

    def __init__(self, speech=None, fail=False):
        self.speech = speech or Speech(samples=tone(10), sample_rate=22_050)
        self.fail = fail
        self.texts = []

    def is_available(self):
        return self.available

    def synthesize(self, text, **kwargs):
        self.texts.append(text)
        if self.fail:
            raise TTSError("boom")
        return self.speech


def test_speaker_plays_with_device_and_waits(tmp_path):
    played = {}
    engine = FakeEngine()

    def play(samples, samplerate, device):
        played.update(samples=samples, samplerate=samplerate, device=device)

    speaker = Speaker(TtsConfig(device_index=3), engine=engine, play=play, wait=lambda: played.update(waited=True))
    speaker.speak("hello")
    assert played["samplerate"] == 22_050
    assert played["device"] == 3
    assert played["waited"] is True
    assert engine.texts == ["hello"]


def test_speaker_speaks_chunks_separately(tmp_path):
    engine = FakeEngine()
    speaker = Speaker(TtsConfig(), engine=engine, play=lambda *a, **k: None, wait=lambda: None)
    speaker.speak_chunks("One. Two. Three.", max_chars=8)
    assert engine.texts == ["One.", "Two.", "Three."]


def test_speaker_available_false_when_engine_unavailable():
    engine = FakeEngine()
    engine.available = False
    assert Speaker(TtsConfig(), engine=engine).available is False


def test_speaker_propagates_tts_errors():
    speaker = Speaker(TtsConfig(), engine=FakeEngine(fail=True), play=lambda *a, **k: None)
    with pytest.raises(TTSError):
        speaker.speak("hello")
