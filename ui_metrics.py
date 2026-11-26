"""Live system telemetry for the Jarvis web UI.

Everything here is synchronous and free of FastAPI so the sampler and the alert
rules can be tested with fake psutil objects.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import time
from typing import Any, Callable, Mapping

DEFAULT_THRESHOLDS: dict[str, float] = {
    "cpu_warn": 80.0,
    "cpu_critical": 90.0,
    "ram_warn": 85.0,
    "gpu_warn": 90.0,
    "disk_warn": 90.0,
}

_GPU_FIELDS = (
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
)
_GPU_QUERY = ",".join(_GPU_FIELDS)


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_gpu() -> dict[str, Any] | None:
    """NVIDIA state through nvidia-smi; None when there is no usable GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    values = [part.strip() for part in line.split(",")]
    if len(values) < len(_GPU_FIELDS):
        return None
    # The name is the only free-text column, so a comma inside it would shift
    # the split. Keep the first token as the name and take the trailing
    # numeric columns by position.
    numbers = values[len(values) - (len(_GPU_FIELDS) - 1):]
    total_mb = _number(numbers[0])
    used_mb = _number(numbers[1])
    return {
        "name": values[0],
        "vram_total_mb": total_mb,
        "vram_used_mb": used_mb,
        "vram_percent": (
            round(used_mb / total_mb * 100, 1)
            if total_mb and used_mb is not None
            else None
        ),
        "util_percent": _number(numbers[2]),
        "temperature_c": _number(numbers[3]),
    }


def evaluate_alerts(
    snapshot: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Pure rules: a snapshot in, the alerts to show out, worst first."""
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update({k: float(v) for k, v in thresholds.items() if v is not None})
    alerts: list[dict[str, Any]] = []

    cpu = (snapshot.get("cpu") or {}).get("percent")
    if cpu is not None:
        if cpu >= limits["cpu_critical"]:
            alerts.append(
                {
                    "key": "cpu",
                    "label": "CPU OVERLOAD",
                    "severity": "critical",
                    "detail": f"{cpu:.0f}% of all cores",
                }
            )
        elif cpu >= limits["cpu_warn"]:
            alerts.append(
                {
                    "key": "cpu",
                    "label": "CPU HIGH",
                    "severity": "warn",
                    "detail": f"{cpu:.0f}% of all cores",
                }
            )

    ram = (snapshot.get("memory") or {}).get("percent")
    if ram is not None and ram >= limits["ram_warn"]:
        alerts.append(
            {
                "key": "ram",
                "label": "RAM HIGH",
                "severity": "warn",
                "detail": f"{ram:.0f}% in use",
            }
        )

    disk = (snapshot.get("disk") or {}).get("percent")
    if disk is not None and disk >= limits["disk_warn"]:
        alerts.append(
            {
                "key": "disk",
                "label": "DISK FULL",
                "severity": "warn",
                "detail": f"{disk:.0f}% used",
            }
        )

    gpu = snapshot.get("gpu")
    if gpu:
        util = gpu.get("util_percent")
        vram = gpu.get("vram_percent")
        if util is not None and util >= limits["gpu_warn"]:
            alerts.append(
                {
                    "key": "gpu",
                    "label": "GPU HIGH",
                    "severity": "warn",
                    "detail": f"{util:.0f}% utilisation",
                }
            )
        if vram is not None and vram >= limits["gpu_warn"]:
            alerts.append(
                {
                    "key": "vram",
                    "label": "VRAM HIGH",
                    "severity": "warn",
                    "detail": f"{vram:.0f}% of VRAM",
                }
            )

    alerts.sort(key=lambda a: 0 if a["severity"] == "critical" else 1)
    return alerts


class MetricsSampler:
    """Turns psutil counters into one JSON-ready snapshot per tick.

    `cpu_percent` and the network counters are deltas since the previous call,
    so the first snapshot reports 0 % and 0 B/s while warming up.
    """

    def __init__(
        self,
        *,
        thresholds: Mapping[str, float] | None = None,
        proc: Any = None,
        gpu_reader: Callable[[], dict[str, Any] | None] = read_gpu,
        clock: Callable[[], float] = time.monotonic,
        disk_path: str | None = None,
    ) -> None:
        if proc is None:
            import psutil

            proc = psutil
        self._proc = proc
        self._gpu_reader = gpu_reader
        self._clock = clock
        self._thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(
                {k: float(v) for k, v in thresholds.items() if v is not None}
            )
        self._disk_path = disk_path or os.path.abspath(os.sep)
        self._last_net: tuple[float, float, float] | None = None

    # -- helpers -------------------------------------------------------- #
    def _safe_uptime(self) -> float:
        try:
            return max(0.0, time.time() - float(self._proc.boot_time()))
        except Exception:  # pragma: no cover - defensive around psutil
            return 0.0

    def _net_rates(self, now: float) -> tuple[float, float, int, int]:
        counters = self._proc.net_io_counters()
        sent, received = int(counters.bytes_sent), int(counters.bytes_recv)
        up = down = 0.0
        if self._last_net is not None:
            last_sent, last_recv, last_at = self._last_net
            elapsed = now - last_at
            if elapsed > 0:
                up = max(0.0, (sent - last_sent) / elapsed)
                down = max(0.0, (received - last_recv) / elapsed)
        self._last_net = (sent, received, now)
        return up, down, sent, received

    # -- public --------------------------------------------------------- #
    def sample(self) -> dict[str, Any]:
        proc = self._proc
        now = self._clock()

        cpu_percent = float(proc.cpu_percent(interval=None))
        try:
            per_core = [float(v) for v in proc.cpu_percent(interval=None, percpu=True)]
        except Exception:  # pragma: no cover - some fakes lack percpu
            per_core = []
        try:
            load = [round(v, 2) for v in os.getloadavg()]
        except (AttributeError, OSError):
            load = None
        freq = None
        try:
            current = proc.cpu_freq()
            freq = round(float(current.current)) if current else None
        except Exception:  # pragma: no cover - Windows can raise here
            freq = None

        memory = proc.virtual_memory()
        disk = proc.disk_usage(self._disk_path)
        up, down, sent, received = self._net_rates(now)

        snapshot: dict[str, Any] = {
            "ts": time.time(),
            "cpu": {
                "percent": round(cpu_percent, 1),
                "per_core": [round(v, 1) for v in per_core],
                "cores": len(per_core) or int(proc.cpu_count() or 0),
                "freq_mhz": freq,
                "load": load,
            },
            "memory": {
                "total": int(memory.total),
                "used": int(memory.used),
                "available": int(memory.available),
                "percent": round(float(memory.percent), 1),
            },
            "disk": {
                "path": self._disk_path,
                "total": int(disk.total),
                "used": int(disk.used),
                "percent": round(float(disk.percent), 1),
            },
            "net": {
                "up_bps": round(up, 1),
                "down_bps": round(down, 1),
                "sent": sent,
                "recv": received,
            },
            "gpu": None,
            "host": {
                "name": socket.gethostname(),
                "platform": platform.system(),
                "python": platform.python_version(),
                "uptime_s": round(self._safe_uptime(), 1),
            },
        }
        try:
            snapshot["gpu"] = self._gpu_reader()
        except Exception:  # pragma: no cover - never let a GPU query kill a tick
            snapshot["gpu"] = None
        snapshot["alerts"] = evaluate_alerts(snapshot, self._thresholds)
        return snapshot
