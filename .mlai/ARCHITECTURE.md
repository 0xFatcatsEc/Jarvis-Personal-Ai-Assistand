# Architecture

Jarvis is one Python process with two front ends over the same core.

```
main.py                 argparse entry point
  ├─ brain.py           LLM provider chain (Ollama / OpenAI-compatible)
  ├─ memory.py          SQLite + Chroma facts and conversation recall
  ├─ tools.py           shell/weather/reminder tools behind ConfirmationRequired
  ├─ audio.py tts.py    wake word -> Whisper, Piper speech
  ├─ ui_server.py       FastAPI: /api/metrics (SSE), /api/chat (WebSocket), /api/state
  │    └─ ui/           index.html + app.js + style.css, no build step
  └─ desktop.py         --app: native window around the same page
```

## Entry points

- `python main.py` - voice loop (wake word, Whisper, Piper).
- `python main.py --text` - typed loop.
- `python main.py --ui` - browser dashboard only; serves `ui/` at `cfg.ui`.
- `python main.py --app` - desktop window: the same dashboard plus the
  microphone, optional `--text` for no mic.

## Desktop shell (desktop.py)

- `DashboardServer` is a thread running **one asyncio loop** that hosts uvicorn
  and `Jarvis.run_voice()` as sibling tasks. One process, one Jarvis instance,
  one memory DB - spoken and typed turns share history and tools.
- `Jarvis.on_voice(role, text)` is awaited around every spoken turn; the
  desktop shell appends the turn to `app.state.ui` and calls
  `ui_server.broadcast()` so every open chat socket receives a `{"type":"turn"}`
  frame. `ui/app.js` renders it in the same conversation panel.
- Window backends, in order: pywebview (Edge WebView2 on Windows), then an
  Edge/Chrome `--app=` window with a throwaway profile, then a clear error with
  install instructions. It never opens a browser tab.
- Tray icon (`pystray`) and the `Ctrl+Alt+J` summon hotkey (`keyboard`) are
  optional; missing packages are logged and skipped.
- `cfg.ui.port` is used unless it is busy, in which case a free port is chosen.

## Verification

- `python -m pytest -q` (Python 3.11 venv; see `.github/workflows`).
- `tests/test_desktop.py` covers port choice, the real server thread serving
  `ui/`, socket broadcast, voice turns landing in UI state, and the window
  fallback chain (with a stubbed webview, since CI is headless).
