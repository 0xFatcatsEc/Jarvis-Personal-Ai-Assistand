"""Tests for desktop.py: window URL, port choice, and the app's server thread."""

from __future__ import annotations

import asyncio
import socket
import types

import httpx

import desktop
import ui_server
from config import Config


class FakeJarvis:
    """Only the attributes the desktop shell touches."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.name = "Jarvis"
        self.on_voice = None
        self.voice_calls = 0

    async def run_voice(self) -> None:
        self.voice_calls += 1


def _cfg(port: int) -> Config:
    return Config(assistant={"name": "Jarvis"}, ui={"port": port, "open_browser": False})


def _start(cfg: Config, jarvis: FakeJarvis) -> tuple[desktop.DashboardServer, str]:
    port = desktop.free_port("127.0.0.1")
    server = desktop.DashboardServer(cfg, jarvis, port=port, voice=False)
    server.start()
    url = desktop.app_url("127.0.0.1", port)
    assert desktop.wait_until_ready(url, timeout=20), "dashboard did not come up"
    return server, url


def test_app_url_reaches_loopback_for_a_wildcard_bind():
    assert desktop.app_url("0.0.0.0", 8765) == "http://127.0.0.1:8765/"
    assert desktop.app_url("::", 8765) == "http://[::1]:8765/"
    assert desktop.app_url("127.0.0.1", 9000) == "http://127.0.0.1:9000/"
    assert desktop.app_url("my-desktop.local", 9000) == "http://my-desktop.local:9000/"


def test_choose_port_moves_when_the_configured_port_is_taken():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        busy = holder.getsockname()[1]
        assert desktop.port_in_use("127.0.0.1", busy)
        chosen = desktop.choose_port("127.0.0.1", busy)
        assert chosen != busy
        assert not desktop.port_in_use("127.0.0.1", chosen)


def test_choose_port_scans_up_from_a_busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        busy = holder.getsockname()[1]
        chosen = desktop.choose_port("127.0.0.1", busy)

        assert chosen != busy
        assert not desktop.port_in_use("127.0.0.1", chosen)
        assert desktop._can_bind("127.0.0.1", chosen)


def test_free_port_skips_a_candidate_that_still_answers(monkeypatch):
    seen = []

    def fake_in_use(host, port):
        seen.append(port)
        return len(seen) == 1

    monkeypatch.setattr(desktop, "port_in_use", fake_in_use)
    desktop.free_port("127.0.0.1")
    assert len(seen) >= 2


def test_choose_port_keeps_a_free_port():
    free = desktop.free_port("127.0.0.1")
    assert desktop.choose_port("127.0.0.1", free) == free


def test_window_serves_the_same_dashboard_page():
    cfg = _cfg(8765)
    jarvis = FakeJarvis(cfg)
    server, url = _start(cfg, jarvis)
    try:
        page = httpx.get(url, timeout=5)
        assert page.status_code == 200
        assert "jarvis" in page.text.lower()
        assert httpx.get(url + "healthz", timeout=5).json()["ok"] is True
        assert jarvis.on_voice is not None, "spoken turns must reach the window"
    finally:
        server.stop()
        server.join(timeout=10)
    assert not server.is_alive()


def test_broadcast_reaches_every_window_and_drops_dead_sockets():
    class Socket:
        def __init__(self, dead: bool = False) -> None:
            self.sent: list[dict] = []
            self.dead = dead

        async def send_json(self, payload: dict) -> None:
            if self.dead:
                raise RuntimeError("window closed")
            self.sent.append(payload)

    good, dead = Socket(), Socket(dead=True)
    app = types.SimpleNamespace(state=types.SimpleNamespace(sockets={good, dead}))
    payload = {"type": "turn", "role": "user", "text": "hello"}

    asyncio.run(ui_server.broadcast(app, payload))

    assert good.sent == [payload]
    assert app.state.sockets == {good}


def test_voice_turn_lands_in_the_conversation_panel_state():
    cfg = Config(assistant={"name": "Jarvis"})
    app = ui_server.create_app(cfg, jarvis=None)
    server = desktop.DashboardServer(cfg, FakeJarvis(cfg), port=1, voice=False)
    server.app = app

    asyncio.run(server._voice_turn("user", "what time is it"))
    assert app.state.ui.state == "thinking"
    assert app.state.ui.turns[-1]["role"] == "user"
    assert app.state.ui.turns[-1]["text"] == "what time is it"

    asyncio.run(server._voice_turn("assistant", "It is noon."))
    assert app.state.ui.state == "idle"
    assert app.state.ui.turn_count == 1
    assert app.state.ui.turns[-1]["role"] == "assistant"
    assert app.state.ui.turns[-1]["text"] == "It is noon."


def test_voice_failure_is_reported_in_the_window():
    cfg = Config(assistant={"name": "Jarvis"})
    app = ui_server.create_app(cfg, jarvis=None)
    jarvis = FakeJarvis(cfg)

    async def no_wake_word() -> None:
        raise RuntimeError("wake word model 'hey_jarvis' is not available")

    jarvis.run_voice = no_wake_word
    server = desktop.DashboardServer(cfg, jarvis, port=1, voice=False)
    server.app = app

    asyncio.run(server._voice())
    assert app.state.ui.state == "idle"
    assert app.state.ui.turns[-1]["text"].startswith("Voice mode unavailable: wake word model")


class FakeWindow:
    def __init__(self, title, url, kwargs):
        self.title = title
        self.url = url
        self.kwargs = kwargs
        self.calls: list[str] = []

    def show(self):
        self.calls.append("show")

    def hide(self):
        self.calls.append("hide")

    def restore(self):
        self.calls.append("restore")

    def destroy(self):
        self.calls.append("destroy")


class FakeWebview:
    def __init__(self, start_error=None):
        self.created: list[FakeWindow] = []
        self.started = False
        self.start_error = start_error

    def create_window(self, title, url, **kwargs):
        window = FakeWindow(title, url, kwargs)
        self.created.append(window)
        return window

    def start(self):
        self.started = True
        if self.start_error is not None:
            raise self.start_error


def test_native_window_hosts_the_running_dashboard(monkeypatch):
    webview = FakeWebview()
    monkeypatch.setattr(desktop, "_load_webview", lambda: webview)

    assert desktop._open_window("http://127.0.0.1:8765/", "Jarvis") == 0

    assert webview.started
    window = webview.created[0]
    assert window.title == "Jarvis"
    assert window.url == "http://127.0.0.1:8765/"
    assert window.kwargs["width"] == desktop.WINDOW_SIZE[0]
    assert window.kwargs["min_size"] == desktop.WINDOW_MIN


def test_failed_native_window_falls_back_to_app_mode(monkeypatch):
    webview = FakeWebview(start_error=RuntimeError("no WebView2"))
    monkeypatch.setattr(desktop, "_load_webview", lambda: webview)
    opened: list[str] = []

    class Process:
        def wait(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(
        desktop, "open_app_window", lambda url: opened.append(url) or Process()
    )

    assert desktop._open_window("http://127.0.0.1:9000/", "Jarvis") == 0
    assert opened == ["http://127.0.0.1:9000/"]


def test_no_window_backend_reports_instead_of_opening_a_browser(monkeypatch):
    monkeypatch.setattr(desktop, "_load_webview", lambda: None)
    monkeypatch.setattr(desktop, "open_app_window", lambda url: None)

    assert desktop._open_window("http://127.0.0.1:9000/", "Jarvis") == 1

