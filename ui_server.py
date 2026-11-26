"""Local web dashboard for Jarvis.

    python main.py --ui      # http://127.0.0.1:8765

The page is served from ui/ with no build step. System telemetry arrives over
Server-Sent Events, chat over one WebSocket so the reply can stream token by
token and the core ring can show its state. Everything binds to localhost by
default; ui.host: 0.0.0.0 exposes it to the LAN without authentication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import Config
from ui_metrics import MetricsSampler

log = logging.getLogger("jarvis.ui")

UI_DIR = Path(__file__).resolve().parent / "ui"
GEO_URL = "https://ipapi.co/json/"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TTL_S = 600.0
LOG_TAIL_BYTES = 32_768
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} "
    r"(?P<level>[A-Z]+) (?P<name>[\w.]+): (?P<msg>.*)$"
)

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Freezing fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Heavy showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with hail",
}


# --------------------------------------------------------------------------- #
# weather
# --------------------------------------------------------------------------- #
class WeatherService:
    """Open-Meteo with a ten minute cache; every failure degrades to None."""

    def __init__(
        self,
        cfg: Config,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        ttl: float = WEATHER_TTL_S,
        clock: Any = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._transport = transport
        self._ttl = ttl
        self._clock = clock
        self._cache: dict[str, Any] | None = None
        self._at = 0.0
        self._lock = asyncio.Lock()

    async def current(self) -> dict[str, Any] | None:
        now = self._clock()
        if self._cache is not None and now - self._at < self._ttl:
            return self._cache
        async with self._lock:
            now = self._clock()
            if self._cache is not None and now - self._at < self._ttl:
                return self._cache
            try:
                data = await self._fetch()
            except Exception as exc:  # network down, blocked, bad payload
                log.info("weather unavailable: %s", exc)
                data = None
            if data is not None:
                self._cache = data
                self._at = now
            return self._cache

    async def _location(self, client: httpx.AsyncClient) -> dict[str, Any]:
        pinned = self._cfg.ui.location
        if pinned.latitude is not None and pinned.longitude is not None:
            return {
                "name": pinned.name or "Pinned location",
                "latitude": pinned.latitude,
                "longitude": pinned.longitude,
            }
        response = await client.get(GEO_URL)
        response.raise_for_status()
        body = response.json()
        latitude, longitude = body.get("latitude"), body.get("longitude")
        if latitude is None or longitude is None:
            raise ValueError("geolocation returned no coordinates")
        return {
            "name": body.get("city") or body.get("region") or "Unknown",
            "latitude": float(latitude),
            "longitude": float(longitude),
        }

    async def _fetch(self) -> dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=6.0, transport=self._transport, headers={"User-Agent": "jarvis-ui/1"}
        ) as client:
            location = await self._location(client)
            response = await client.get(
                WEATHER_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": ",".join(
                        [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "apparent_temperature",
                            "weather_code",
                            "wind_speed_10m",
                            "is_day",
                        ]
                    ),
                    "timezone": "auto",
                },
            )
            response.raise_for_status()
            current = response.json().get("current") or {}
        code = int(current.get("weather_code") or 0)
        return {
            **location,
            "temperature_c": current.get("temperature_2m"),
            "apparent_c": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_kph": current.get("wind_speed_10m"),
            "is_day": bool(current.get("is_day", 1)),
            "code": code,
            "condition": WEATHER_CODES.get(code, "Unknown"),
            "updated": time.time(),
        }


# --------------------------------------------------------------------------- #
# logs
# --------------------------------------------------------------------------- #
def read_log_tail(path: Path, *, limit: int = 14) -> list[dict[str, Any]]:
    """Last parsed lines of jarvis.log, oldest first; [] when it is missing."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - LOG_TAIL_BYTES))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if size > LOG_TAIL_BYTES and lines:
        lines = lines[1:]  # first line is very likely cut in half
    entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        match = LOG_LINE_RE.match(line)
        if match:
            entries.append(
                {
                    "ts": match.group("ts"),
                    "time": match.group("ts")[11:],
                    "level": match.group("level"),
                    "source": match.group("name"),
                    "message": match.group("msg"),
                    "raw": line,
                }
            )
        else:
            entries.append(
                {
                    "ts": "",
                    "time": time.strftime("%H:%M:%S"),
                    "level": "RAW",
                    "source": "",
                    "message": line,
                    "raw": line,
                }
            )
    return entries


# --------------------------------------------------------------------------- #
# shared state
# --------------------------------------------------------------------------- #
class UiState:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = "idle"
        self.detail = "Standing by"
        self.turns: deque[dict[str, Any]] = deque(maxlen=80)
        self.turn_count = 0
        self.last_latency_ms: int | None = None
        self.started = time.time()

    def core(self) -> dict[str, Any]:
        provider = next((p for p in self.cfg.routing_order() if p.usable), None)
        return {
            "state": self.state,
            "detail": self.detail,
            "turns": self.turn_count,
            "last_latency_ms": self.last_latency_ms,
            "uptime_s": round(time.time() - self.started, 1),
            "model": provider.model if provider else None,
            "provider": provider.name if provider else None,
            "routing": self.cfg.llm.routing.mode,
            "context": (provider.options or {}).get("num_ctx") if provider else None,
            "name": self.cfg.assistant.name,
        }


def create_app(
    cfg: Config,
    jarvis: Any | None = None,
    *,
    weather: WeatherService | None = None,
) -> FastAPI:
    app = FastAPI(title="Jarvis UI", docs_url=None, redoc_url=None, openapi_url=None)
    state = UiState(cfg)
    sampler = MetricsSampler(
        thresholds=cfg.ui.alerts.model_dump(),
        disk_path=str(cfg.assistant.data_dir.resolve()),
    )
    weather_service = weather or WeatherService(cfg)
    chat_lock = asyncio.Lock()
    sockets: set[WebSocket] = set()
    log_path = cfg.assistant.log_dir / "jarvis.log"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "state": state.state})

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(
            {
                "core": state.core(),
                "turns": list(state.turns),
                "ui": {
                    "refresh_ms": cfg.ui.refresh_ms,
                    "host": cfg.ui.host,
                    "port": cfg.ui.port,
                },
                "thresholds": cfg.ui.alerts.model_dump(),
                "location": cfg.ui.location.model_dump(),
            }
        )

    @app.get("/api/metrics")
    async def metrics() -> StreamingResponse:
        async def events() -> AsyncIterator[bytes]:
            while True:
                payload = {
                    "system": await asyncio.to_thread(sampler.sample),
                    "core": state.core(),
                    "weather": await weather_service.current(),
                    "activity": await asyncio.to_thread(read_log_tail, log_path),
                    "now": time.time(),
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
                await asyncio.sleep(cfg.ui.refresh_ms / 1000)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.websocket("/api/chat")
    async def chat(socket: WebSocket) -> None:
        await socket.accept()
        if jarvis is None:
            await socket.send_json({"type": "error", "text": "Jarvis core is not running."})
            await socket.close()
            return
        sockets.add(socket)
        try:
            while True:
                message = await socket.receive_json()
                text = " ".join(str((message or {}).get("text") or "").split())
                if not text:
                    continue
                await run_turn(socket, text)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - a dead socket must not kill the server
            log.exception("chat socket failed")
        finally:
            sockets.discard(socket)

    async def run_turn(socket: WebSocket, text: str) -> None:
        if chat_lock.locked():
            await socket.send_json(
                {"type": "error", "text": "Jarvis is still answering the previous message."}
            )
            return
        async with chat_lock:
            state.state = "thinking"
            state.detail = "Processing your request"
            state.turns.append({"role": "user", "text": text, "at": time.time()})
            await socket.send_json({"type": "status", "state": "thinking", "text": state.detail})

            async def on_delta(chunk: str) -> None:
                try:
                    await socket.send_json({"type": "delta", "text": chunk})
                except Exception:  # client vanished mid-stream
                    pass

            started = time.monotonic()
            jarvis.on_delta = on_delta
            try:
                reply = await jarvis.handle(text, speak=False)
                latency = int((time.monotonic() - started) * 1000)
                state.state = "idle"
                state.detail = "Standing by"
                state.last_latency_ms = latency
                state.turn_count += 1
                state.turns.append({"role": "assistant", "text": reply, "at": time.time()})
                await socket.send_json(
                    {"type": "reply", "text": reply, "latency_ms": latency}
                )
            except Exception as exc:  # noqa: BLE001 - report, never crash the server
                log.exception("ui turn failed")
                state.state = "error"
                state.detail = "Fault detected"
                await socket.send_json({"type": "error", "text": str(exc) or "Jarvis failed."})
            finally:
                jarvis.on_delta = None

    if UI_DIR.is_dir():
        # Mounted last so the API routes above win; html=True serves index.html
        # at "/" and the assets next to it (style.css, app.js).
        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")

    app.state.ui = state
    app.state.sampler = sampler
    app.state.sockets = sockets
    return app


async def broadcast(app: FastAPI, payload: dict[str, Any]) -> None:
    """Push a frame to every open chat socket (spoken turns, for the window)."""
    for socket in list(getattr(app.state, "sockets", ())):
        try:
            await socket.send_json(payload)
        except Exception:  # noqa: BLE001 - a stale window must not break the turn
            app.state.sockets.discard(socket)


# --------------------------------------------------------------------------- #
# serving
# --------------------------------------------------------------------------- #
async def serve_ui(jarvis: Any) -> None:
    import uvicorn

    cfg: Config = jarvis.cfg
    app = create_app(cfg, jarvis)
    url_host = "127.0.0.1" if cfg.ui.host == "0.0.0.0" else cfg.ui.host
    url = f"http://{url_host}:{cfg.ui.port}/"
    print(f"Jarvis UI on {url}  (Ctrl+C to stop)", flush=True)
    if cfg.ui.host == "0.0.0.0":
        print("warning: bound to 0.0.0.0 - anyone on your network can chat with Jarvis", flush=True)

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=cfg.ui.host,
            port=cfg.ui.port,
            log_level="warning",
            access_log=False,
        )
    )
    opener = None
    if cfg.ui.open_browser:
        opener = asyncio.create_task(_open_browser(url))
    try:
        await server.serve()
    finally:
        if opener is not None:
            opener.cancel()


async def _open_browser(url: str) -> None:
    await asyncio.sleep(1.2)
    try:
        await asyncio.to_thread(webbrowser.open, url)
    except Exception:  # headless box: nothing to open
        pass
