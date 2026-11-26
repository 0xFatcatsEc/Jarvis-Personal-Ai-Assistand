"""Tests for doctor.py: voice URLs/download, device listing, checks, exit codes."""

from __future__ import annotations

import httpx
import pytest

import doctor
from config import Config, LlmConfig, ProviderConfig, TtsConfig


def make_config(tmp_path, **tts) -> Config:
    return Config(tts=TtsConfig(voices_dir=tmp_path, **tts))


def test_voice_urls_point_at_huggingface():
    onnx, meta = doctor.voice_urls("en_US-lessac-medium")
    stem = "en/en_US/lessac/medium/en_US-lessac-medium"
    assert onnx == f"{doctor.VOICE_BASE}/{stem}.onnx"
    assert meta == f"{doctor.VOICE_BASE}/{stem}.onnx.json"


@pytest.mark.parametrize("voice", ["lessac", "en_US-lessac", "en_US-lessac-medium-high"])
def test_voice_urls_rejects_unknown_shape(voice):
    with pytest.raises(doctor.DoctorError):
        doctor.voice_urls(voice)


def test_download_voice_fetches_both_files(tmp_path):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        body = b"ONNXBYTES" if request.url.path.endswith(".onnx") else b"{}"
        return httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = make_config(tmp_path)
    written = doctor.download_voice(cfg, client=client, log=lambda line: None)

    onnx = tmp_path / "en_US-lessac-medium.onnx"
    meta = tmp_path / "en_US-lessac-medium.onnx.json"
    assert onnx.read_bytes() == b"ONNXBYTES"
    assert meta.read_bytes() == b"{}"
    assert set(written) == {onnx, meta}
    assert len(requested) == 2

    doctor.download_voice(cfg, client=client, log=lambda line: None)
    assert len(requested) == 2


def test_download_voice_reports_http_errors(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with pytest.raises(doctor.DoctorError, match="download failed"):
        doctor.download_voice(make_config(tmp_path), client=client, log=lambda line: None)
    assert not (tmp_path / "en_US-lessac-medium.onnx").exists()


def test_download_voice_refuses_custom_model_path(tmp_path):
    cfg = make_config(tmp_path, model_path=tmp_path / "custom.onnx")
    with pytest.raises(doctor.DoctorError, match="model_path"):
        doctor.download_voice(cfg, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))))


def test_check_voice_reports_missing_without_download(tmp_path):
    check = doctor.check_voice(make_config(tmp_path), download=False)
    assert check.status == "warn" and "missing" in check.detail


def test_check_voice_ok_when_file_exists(tmp_path):
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"x")
    assert doctor.check_voice(make_config(tmp_path), download=False).status == "ok"


class FakeSd:
    def __init__(self, devices):
        self.devices = devices

    def query_devices(self):
        return self.devices


def test_audio_devices_separates_inputs_and_outputs():
    sd = FakeSd(
        [
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Yeti", "max_input_channels": 1, "max_output_channels": 0},
        ]
    )
    inputs, outputs = doctor.audio_devices(sd)
    assert inputs == ["Yeti"] and outputs == ["Speakers"]


def test_check_audio_warns_without_microphone():
    check = doctor.check_audio(FakeSd([{"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2}]))
    assert check.status == "warn" and "no microphone" in check.detail


def test_check_audio_ok_with_devices():
    check = doctor.check_audio(FakeSd([{"name": "Yeti", "max_input_channels": 1, "max_output_channels": 2}]))
    assert check.status == "ok" and "Yeti" in check.detail


def test_check_ollama_reports_available_model():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert doctor.check_ollama(Config(), client=client).status == "ok"


def test_check_ollama_warns_when_model_missing():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [{"name": "mistral:7b"}]}))
    )
    check = doctor.check_ollama(Config(), client=client)
    assert check.status == "warn" and "not pulled" in check.detail


def test_check_ollama_warns_when_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    check = doctor.check_ollama(Config(), client=client)
    assert check.status == "warn" and "unreachable" in check.detail


def test_check_keys_lists_unset_secrets():
    cfg = Config(
        llm=LlmConfig(
            providers=[
                ProviderConfig(name="ollama", model="llama3"),
                ProviderConfig(name="openai", model="gpt-4o-mini"),
            ]
        )
    )
    check = doctor.check_keys(cfg)
    assert check.status == "warn"
    assert "OPENAI_API_KEY" in check.detail
    assert "HOME_ASSISTANT_TOKEN" in check.detail


def test_summarize_exit_codes():
    assert doctor.summarize([doctor.Check("a", "ok", "")]) == 0
    assert doctor.summarize([doctor.Check("a", "ok", ""), doctor.Check("b", "warn", "")]) == 1
    assert doctor.summarize([doctor.Check("a", "fail", "")]) == 1


def test_check_audio_warns_when_portaudio_cannot_initialize():
    class BrokenSd:
        def query_devices(self):
            raise RuntimeError("PulseAudio: Can't connect to server")

    check = doctor.check_audio(BrokenSd())
    assert check.status == "warn" and "cannot list audio devices" in check.detail


def test_cli_flags():
    args = doctor.parse_args(["--no-download", "--json"])
    assert args.no_download is True and args.json is True
    assert doctor.parse_args([]).no_download is False
