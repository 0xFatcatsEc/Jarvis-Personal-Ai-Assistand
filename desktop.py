"""Desktop window for Jarvis: the dashboard, in its own application window.

    python main.py --app          # window + microphone (wake word voice mode)
    python main.py --app --text   # window only, no microphone

The page is the same ui/ dashboard `--ui` serves; only the frame changes. The
HTTP server and the voice loop share one asyncio loop in a worker thread, so
spoken and typed input reach the same Jarvis, the same memory and the same
tools, and a spoken turn appears in the window's conversation panel.

The window itself is pywebview (Edge WebView2 on Windows). When that backend
is not installed or cannot start, the page opens in an Edge/Chrome app-mode
window instead - also a chromeless window with its own taskbar entry - so a
missing backend never drops Jarvis back into a browser tab.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("jarvis.desktop")

WINDOW_TITLE = "Jarvis"
WINDOW_SIZE = (1440, 900)
WINDOW_MIN = (1100, 700)
WINDOW_BG = "#05090d"
READY_TIMEOUT_S = 30.0
HOTKEY = "ctrl+alt+j"

_BROWSER_PATHS = {
    "win32": (
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ),
    "darwin": (
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ),
    "linux": ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"),
}


# --------------------------------------------------------------------------- #
# address helpers
# --------------------------------------------------------------------------- #
def connect_host(host: str) -> str:
    """A bind address turned into one a client can connect to.

    A wildcard bind (0.0.0.0, ::) is not a destination, so it resolves to the
    matching loopback address; hostnames and specific addresses pass through.
    """
    if not host:
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.is_unspecified:
        return "::1" if address.version == 6 else "127.0.0.1"
    return host


def app_url(host: str, port: int) -> str:
    """The URL the window loads."""
    target = connect_host(host)
    if ":" in target:  # IPv6 literals need brackets in a URL
        target = f"[{target}]"
    return f"http://{target}:{port}/"


def port_in_use(host: str, port: int) -> bool:
    """Whether the port is taken, decided by trying to bind it.

    Windows' connect probe is wrong in both directions here (it accepted a
    port a socket was already listening on, and refused one we had just bound),
    while an exclusive bind answers the question the server actually asks: can
    this process listen on it?
    """
    if port <= 0:
        return False
    return not _can_bind(host, port)


def _can_bind(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "nt":
            # Without this Windows happily binds a port another socket holds.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def free_port(host: str, *, near: int = 0, span: int = 64) -> int:
    """A port nothing holds, preferring numbers just above `near`.

    Free means an exclusive bind succeeded - the one check Windows cannot fake
    by handing back a port another socket is already listening on.
    """
    if near > 0:
        for port in range(near + 1, near + 1 + span):
            if _can_bind(host, port):
                return port
    port = 0
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if os.name == "nt":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])
        if _can_bind(host, port):
            return port
        log.debug("free_port: %s cannot bind %s; asking again", host, port)
    return port


def choose_port(host: str, preferred: int) -> int:
    """Use the configured port, unless another process already holds it."""
    if preferred > 0 and _can_bind(host, preferred):
        return preferred
    chosen = free_port(host, near=preferred)
    log.info("ui port %s is busy; using %s for this window", preferred, chosen)
    return chosen


def wait_until_ready(
    url: str, timeout: float = READY_TIMEOUT_S, *, still_running: Any = None
) -> bool:
    """Poll /healthz; give up early when the server thread has already died."""
    health = url.rstrip("/") + "/healthz"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if still_running is not None and not still_running():
            return False
        try:
            if httpx.get(health, timeout=1.0).is_success:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.15)
    return False


# --------------------------------------------------------------------------- #
# server thread: dashboard + voice on one event loop
# --------------------------------------------------------------------------- #
class DashboardServer(threading.Thread):
    def __init__(self, cfg: Any, jarvis: Any, *, port: int, voice: bool = True) -> None:
        super().__init__(name="jarvis-desktop", daemon=True)
        self.cfg = cfg
        self.jarvis = jarvis
        self.port = port
        self.voice = voice
        self.app: Any = None
        self.error: BaseException | None = None
        self._server: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # noqa: BLE001 - a dead thread must be reported
            self.error = exc
            log.exception("desktop server stopped: %s", exc)

    async def _serve(self) -> None:
        import uvicorn

        from ui_server import create_app

        self._loop = asyncio.get_running_loop()
        self.app = create_app(self.cfg, self.jarvis)
        self.jarvis.on_voice = self._voice_turn
        server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host=self.cfg.ui.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        )
        self._server = server
        tasks = [asyncio.create_task(server.serve(), name="dashboard")]
        if self.voice:
            tasks.append(asyncio.create_task(self._voice(), name="voice"))
        await asyncio.gather(*tasks)

    async def _voice(self) -> None:
        try:
            await self.jarvis.run_voice()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - no mic must not close the window
            reason = str(exc) or exc.__class__.__name__
            log.error("voice mode unavailable (%s); the window still handles typing", exc)
            await self._voice_turn("assistant", f"Voice mode unavailable: {reason}. Typing still works.")

    async def _voice_turn(self, role: str, text: str) -> None:
        from ui_server import broadcast

        app = self.app
        if app is None or not text:
            return
        state = app.state.ui
        if role == "user":
            state.state = "thinking"
            state.detail = "Voice input"
            state.turns.append({"role": "user", "text": text, "at": time.time()})
            print(f"You (voice): {text}", flush=True)
        else:
            state.state = "idle"
            state.detail = "Standing by"
            state.turn_count += 1
            state.turns.append({"role": "assistant", "text": text, "at": time.time()})
        await broadcast(app, {"type": "turn", "role": role, "text": text})

    def stop(self) -> None:
        server, loop = self._server, self._loop
        if server is None or loop is None:
            return
        loop.call_soon_threadsafe(setattr, server, "should_exit", True)


# --------------------------------------------------------------------------- #
# window backends
# --------------------------------------------------------------------------- #
def _load_webview() -> Any | None:
    try:
        import webview
    except Exception as exc:  # noqa: BLE001 - optional dependency
        log.info("pywebview unavailable (%s)", exc)
        return None
    return webview


def _window_call(window: Any, method: str, *args: Any) -> None:
    try:
        getattr(window, method)(*args)
    except Exception as exc:  # noqa: BLE001 - the window may already be gone
        log.debug("window.%s failed: %s", method, exc)


def _tray_image() -> Any | None:
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    image = Image.new("RGBA", (64, 64), (5, 9, 13, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), outline=(34, 227, 255, 255), width=4)
    draw.ellipse((26, 26, 38, 38), fill=(34, 227, 255, 255))
    return image


def _start_tray(window: Any, name: str) -> tuple[Any, Any] | None:
    try:
        import pystray
    except Exception as exc:  # noqa: BLE001 - optional dependency
        log.info("tray icon unavailable (%s)", exc)
        return None
    image = _tray_image()
    if image is None:
        return None

    def show(*_: Any) -> None:
        _window_call(window, "show")
        _window_call(window, "restore")

    menu = pystray.Menu(
        pystray.MenuItem("Show Jarvis", show, default=True),
        pystray.MenuItem("Hide", lambda *_: _window_call(window, "hide")),
        pystray.MenuItem("Quit", lambda *_: _window_call(window, "destroy")),
    )
    icon = pystray.Icon("jarvis", image, name, menu)
    thread = threading.Thread(target=icon.run, name="jarvis-tray", daemon=True)
    thread.start()
    return icon, thread


def _start_hotkey(window: Any) -> Any | None:
    try:
        import keyboard
    except Exception as exc:  # noqa: BLE001 - optional dependency
        log.info("global hotkey unavailable (%s)", exc)
        return None

    def summon() -> None:
        _window_call(window, "show")
        _window_call(window, "restore")

    try:
        return keyboard.add_hotkey(HOTKEY, summon)
    except Exception as exc:  # noqa: BLE001 - needs root on Linux
        log.info("global hotkey could not be registered (%s)", exc)
        return None


def _stop_extras(tray: tuple[Any, Any] | None, hotkey: Any) -> None:
    if tray is not None:
        icon, _thread = tray
        try:
            icon.stop()
        except Exception:  # noqa: BLE001
            pass
    if hotkey is not None:
        try:
            import keyboard

            keyboard.remove_hotkey(hotkey)
        except Exception:  # noqa: BLE001
            pass


def _run_webview(webview: Any, url: str, title: str) -> int:
    window = webview.create_window(
        title,
        url,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN,
        background_color=WINDOW_BG,
    )
    tray = _start_tray(window, title)
    hotkey = _start_hotkey(window)
    try:
        webview.start()
    finally:
        _stop_extras(tray, hotkey)
    return 0


def find_app_browser() -> str | None:
    for candidate in _BROWSER_PATHS.get(sys.platform, _BROWSER_PATHS["linux"]):
        found = shutil.which(candidate)
        if found:
            return found
        if Path(candidate).is_file():
            return candidate
    return None


def open_app_window(url: str) -> subprocess.Popen[bytes] | None:
    """Chromeless Edge/Chrome window, with a throwaway profile so it never
    merges into an already running browser and opens as a tab instead."""
    exe = find_app_browser()
    if exe is None:
        return None
    profile = tempfile.mkdtemp(prefix="jarvis-app-")
    flags = [
        f"--app={url}",
        f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    try:
        return subprocess.Popen([exe, *flags])
    except OSError as exc:
        log.warning("could not start %s (%s)", exe, exc)
        shutil.rmtree(profile, ignore_errors=True)
        return None


def _open_window(url: str, title: str) -> int:
    webview = _load_webview()
    if webview is not None:
        try:
            return _run_webview(webview, url, title)
        except Exception as exc:  # noqa: BLE001 - fall through to app mode
            log.warning("native window failed (%s); using an app-mode window", exc)

    process = open_app_window(url)
    if process is not None:
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
        return 0

    print(
        "No desktop window could start. Install the window backend with\n"
        "    pip install -r requirements.txt\n"
        "or use the browser dashboard: python main.py --ui",
        file=sys.stderr,
        flush=True,
    )
    return 1


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run_app(cfg: Any, jarvis: Any, *, voice: bool = True) -> int:
    host = "127.0.0.1" if cfg.ui.host == "0.0.0.0" else cfg.ui.host
    port = choose_port(host, cfg.ui.port)
    server = DashboardServer(cfg, jarvis, port=port, voice=voice)
    server.start()

    url = app_url(host, port)
    if not wait_until_ready(url, still_running=server.is_alive):
        server.stop()
        server.join(timeout=5)
        reason = server.error or "the dashboard did not answer in time"
        print(f"Jarvis could not start the dashboard on {url}: {reason}", file=sys.stderr, flush=True)
        return 1

    print(f"{cfg.assistant.name} window on {url}", flush=True)
    if voice:
        print("say the wake word after the microphone starts; typed chat works too", flush=True)
    try:
        return _open_window(url, cfg.assistant.name or WINDOW_TITLE)
    finally:
        server.stop()
        server.join(timeout=5)
        if server.error is not None:
            log.debug("server exit: %s", server.error)
