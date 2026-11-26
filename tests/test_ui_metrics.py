"""Tests for ui_metrics.py and the weather cache in ui_server.py."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from config import Config, ConfigError, UiConfig, load_config
import ui_server
from ui_metrics import (
    DEFAULT_THRESHOLDS,
    MetricsSampler,
    evaluate_alerts,
    read_gpu,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeCounters:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeProc:
    """The slice of psutil the sampler touches, driven by plain numbers."""

    def __init__(self, cpu=10.0, per_core=(10.0, 20.0), mem_percent=40.0):
        self.cpu = cpu
        self.per_core = list(per_core)
        self.mem_percent = mem_percent
        self.net = FakeCounters(bytes_sent=0, bytes_recv=0)
        self.calls = 0

    def cpu_percent(self, interval=None, percpu=False):
        if percpu:
            return self.per_core
        self.calls += 1
        return self.cpu

    def cpu_count(self):
        return len(self.per_core)

    def cpu_freq(self):
        return FakeCounters(current=2400.0)

    def virtual_memory(self):
        total = 8 * 1024**3
        return FakeCounters(
            total=total,
            used=int(total * self.mem_percent / 100),
            available=total - int(total * self.mem_percent / 100),
            percent=self.mem_percent,
        )

    def disk_usage(self, path):
        return FakeCounters(total=500 * 1024**3, used=100 * 1024**3, percent=20.0)

    def net_io_counters(self):
        return self.net

    def boot_time(self):
        return 0.0


def snapshot(cpu=None, ram=40.0, gpu=None, disk=20.0):
    return {
        "cpu": {"percent": cpu},
        "memory": {"percent": ram},
        "disk": {"percent": disk},
        "gpu": gpu,
    }


# --------------------------------------------------------------------------- #
# alert rules
# --------------------------------------------------------------------------- #
def test_alerts_quiet_below_thresholds():
    assert evaluate_alerts(snapshot(cpu=10.0)) == []


def test_cpu_warn_then_critical():
    warn = evaluate_alerts(snapshot(cpu=85.0))
    assert [a["label"] for a in warn] == ["CPU HIGH"]
    assert warn[0]["severity"] == "warn"

    critical = evaluate_alerts(snapshot(cpu=97.0))
    assert [a["label"] for a in critical] == ["CPU OVERLOAD"]
    assert critical[0]["severity"] == "critical"


def test_threshold_boundary_is_inclusive():
    assert evaluate_alerts(snapshot(cpu=80.0))[0]["label"] == "CPU HIGH"


def test_ram_disk_and_gpu_rules():
    alerts = evaluate_alerts(
        snapshot(cpu=5.0, ram=91.0, disk=95.0, gpu={"util_percent": 96.0, "vram_percent": 50.0})
    )
    labels = {a["label"] for a in alerts}
    assert labels == {"RAM HIGH", "DISK FULL", "GPU HIGH"}
    assert "VRAM HIGH" not in labels


def test_vram_rule_and_missing_gpu():
    gpu = {"util_percent": 5.0, "vram_percent": 93.0}
    assert [a["key"] for a in evaluate_alerts(snapshot(cpu=1.0, gpu=gpu))] == ["vram"]
    assert evaluate_alerts(snapshot(cpu=1.0, gpu=None)) == []


def test_custom_thresholds_override_defaults():
    alerts = evaluate_alerts(snapshot(cpu=70.0), {"cpu_warn": 60.0, "cpu_critical": 65.0})
    assert alerts[0]["severity"] == "critical"


def test_critical_sorted_first():
    alerts = evaluate_alerts(snapshot(cpu=95.0, ram=90.0))
    assert alerts[0]["severity"] == "critical"


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #
def test_sampler_cold_start_reports_zero_rates():
    proc = FakeProc(cpu=12.0)
    sampler = MetricsSampler(proc=proc, gpu_reader=lambda: None, clock=lambda: 100.0)
    result = sampler.sample()

    assert result["cpu"]["percent"] == 12.0
    assert result["cpu"]["cores"] == 2
    assert result["net"]["up_bps"] == 0.0
    assert result["disk"]["percent"] == 20.0
    assert result["alerts"] == []


def test_sampler_computes_network_rates_between_ticks():
    proc = FakeProc()
    times = iter([100.0, 102.0])
    sampler = MetricsSampler(proc=proc, gpu_reader=lambda: None, clock=lambda: next(times))

    sampler.sample()
    proc.net = FakeCounters(bytes_sent=2048, bytes_recv=4096)
    result = sampler.sample()

    assert result["net"]["up_bps"] == 1024.0
    assert result["net"]["down_bps"] == 2048.0


def test_sampler_uses_gpu_reader_and_thresholds():
    proc = FakeProc(cpu=95.0)
    sampler = MetricsSampler(
        proc=proc,
        gpu_reader=lambda: {
            "name": "RTX 3050",
            "vram_total_mb": 4096.0,
            "vram_used_mb": 2048.0,
            "vram_percent": 50.0,
            "util_percent": 5.0,
            "temperature_c": 55.0,
        },
        clock=lambda: 5.0,
    )
    result = sampler.sample()

    assert result["gpu"]["name"] == "RTX 3050"
    assert [a["label"] for a in result["alerts"]] == ["CPU OVERLOAD"]


def test_sampler_survives_a_broken_gpu_reader():
    def explode():
        raise OSError("nvidia-smi disappeared")

    sampler = MetricsSampler(proc=FakeProc(), gpu_reader=explode, clock=lambda: 1.0)
    assert sampler.sample()["gpu"] is None


def test_read_gpu_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr("ui_metrics.shutil.which", lambda name: None)
    assert read_gpu() is None


def test_read_gpu_parses_nvidia_smi_output(monkeypatch):
    monkeypatch.setattr("ui_metrics.shutil.which", lambda name: "/usr/bin/nvidia-smi")

    class Done:
        stdout = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 1024, 40, 55\n"

    monkeypatch.setattr("ui_metrics.subprocess.run", lambda *a, **k: Done())
    gpu = read_gpu()

    assert gpu["name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert gpu["vram_total_mb"] == 4096.0
    assert gpu["vram_percent"] == 25.0
    assert gpu["util_percent"] == 40.0
    assert gpu["temperature_c"] == 55.0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_ui_defaults_are_local_and_sane():
    ui = UiConfig()
    assert ui.host == "127.0.0.1"
    assert ui.port == 8765
    assert ui.alerts.cpu_critical > ui.alerts.cpu_warn


def test_ui_rejects_half_a_location():
    with pytest.raises(Exception):
        UiConfig.model_validate(
            {"location": {"name": "Manila", "latitude": 14.6}}
        )


def test_ui_rejects_misordered_cpu_thresholds():
    with pytest.raises(Exception):
        UiConfig.model_validate({"alerts": {"cpu_warn": 90, "cpu_critical": 85}})


def test_shipped_config_has_the_ui_section():
    cfg = load_config(REPO / "config.yaml")
    assert cfg.ui.port == 8765
    assert set(DEFAULT_THRESHOLDS) <= set(cfg.ui.alerts.model_dump())


def test_shipped_config_rejects_negative_refresh(tmp_path):
    text = (REPO / "config.yaml").read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(text.replace("refresh_ms: 1500", "refresh_ms: 10"), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


# --------------------------------------------------------------------------- #
# weather
# --------------------------------------------------------------------------- #
def test_weather_uses_configured_location_and_caches():
    cfg = Config(
        assistant={"name": "Jarvis"},
        ui={"location": {"name": "Manila", "latitude": 14.6, "longitude": 120.98}},
    )
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "open-meteo" in request.url.host:
            return httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 28.7,
                        "relative_humidity_2m": 74,
                        "apparent_temperature": 33.1,
                        "weather_code": 2,
                        "wind_speed_10m": 9.4,
                        "is_day": 1,
                    }
                },
            )
        raise AssertionError("pinned coordinates must not call the geolocator")

    clock = [0.0]
    weather = ui_server.WeatherService(
        cfg,
        transport=httpx.MockTransport(handler),
        ttl=600,
        clock=lambda: clock[0],
    )

    first = asyncio.run(weather.current())
    assert first["name"] == "Manila"
    assert first["temperature_c"] == 28.7
    assert first["condition"] == "Partly cloudy"
    assert len(calls) == 1

    clock[0] = 100.0
    asyncio.run(weather.current())
    assert len(calls) == 1  # cache hit inside the TTL


def test_weather_returns_none_when_offline():
    cfg = Config(assistant={"name": "Jarvis"})

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    weather = ui_server.WeatherService(cfg, transport=httpx.MockTransport(handler))
    assert asyncio.run(weather.current()) is None
