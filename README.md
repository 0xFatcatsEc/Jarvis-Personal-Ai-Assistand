# Jarvis

A local-first voice assistant: wake word, speech-to-text, an LLM brain, and a
memory that actually remembers you. Everything runs on your machine; cloud LLM
providers are an optional fallback, never a requirement.

```
wake word -> STT -> recall facts + past turns -> LLM (tools) -> Piper TTS
```

## Status

All seven modules are written and tested (201 tests, headless: no microphone,
Ollama or Piper binary needed). Voice I/O itself still has to be exercised on
the machine with a sound card.

| # | Module | Purpose | Status |
|---|--------|---------|--------|
| 1 | `config.yaml`, `config.py` | single config + validated schema | done |
| 2 | `audio.py` | mic capture, wake word, VAD, faster-whisper STT | done |
| 3 | `brain.py` | streaming LLM router, local -> cloud fallback | done |
| 4 | `memory.py` | SQLite facts + Chroma recall, learn, decay | done |
| 5 | `tools.py` | web search, notes, reminders, Home Assistant, shell, apps | done |
| 6 | `tts.py` | Piper speech out, sentence chunking, playback | done |
| 7 | `main.py` | async loop, tool calls, confirmations, reminders | done |

## Requirements

- **Python 3.11** (pinned in `.python-version`). openWakeWord depends on
  `tflite-runtime`, whose Linux wheels stop at cp311, so 3.12+ cannot install
  the audio stack. `config.py` on its own runs fine on 3.12-3.14.
- **PortAudio** for microphone/speaker access: `sudo apt-get install -y
  libportaudio2` on Debian/Ubuntu, `brew install portaudio` on macOS.
- [Ollama](https://ollama.com) running locally with a model pulled, e.g.
  `ollama pull llama3.1:8b`
- A microphone and speakers, with a working sound server (not available on
  headless machines, so audio runs on the machine you actually speak to)
- Optional: a Piper voice (`en_US-lessac-medium.onnx` + `.json`) in `models/piper/`

## Quickstart

```bash
sudo apt-get install -y libportaudio2    # macOS: brew install portaudio
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + pytest
cp env.example .env                     # optional: cloud keys, HA token

python config.py --check                 # validate config, print what resolved
python doctor.py                         # preflight: voice, audio, Ollama, keys
python main.py --check                   # same check, from the entry point
python memory_cli.py list                # inspect/edit learned facts
python -m pytest                         # tests (201)
```

`memory_cli.py` (`list`, `get KEY`, `set KEY VALUE`, `forget KEY`, `decay`)
reads the same database the agent uses, so you can correct a wrong fact without
editing SQL. Stale facts are shown with a `(stale)` marker.

CI runs on every push and pull request via `.github/workflows/ci.yml`: one job
runs the test suite on Python 3.11 with minimal deps, another installs the full
`requirements.txt` and imports every module.

## Running

```bash
python main.py              # voice: wake word -> speak -> reply
python main.py --app        # desktop window: dashboard + microphone
python main.py --app --text # desktop window, typed chat only
python main.py --text       # typed conversation, no microphone
python main.py --once "hi"  # one exchange, then exit
python main.py --ui         # web dashboard at http://127.0.0.1:8765
```

`--app` is the same dashboard in its own application window: pywebview (Edge
WebView2 on Windows) with a tray icon, `Ctrl+Alt+J` to summon it, and the
microphone running alongside - spoken turns appear in the window's
conversation panel and typed turns are answered silently. If pywebview cannot
start, the app falls back to a chromeless Edge/Chrome app-mode window, so it
never opens as a browser tab. `--app --text` skips the microphone.

The dashboard (`--ui`) shows live CPU/memory/disk/network, GPU VRAM through
`nvidia-smi`, weather with your location, the recent log and a streaming chat,
all from the same `Jarvis.handle` path as `--text`. It binds to localhost by
default; `ui.host: 0.0.0.0` in `config.yaml` exposes it to your LAN with no
authentication, so only do that on a network you trust. It needs `fastapi`,
`uvicorn` and `psutil` from `requirements.txt`.

Before the first voice run, `python doctor.py` checks the environment and
downloads the configured Piper voice (~63 MB) into `models/piper/`. It reports
missing audio devices, an unreachable Ollama, untrained models and unset keys;
`--no-download` skips the fetch and `--json` is for scripts. Exit codes: 0 ready,
1 warnings, 2 bad config.

Two behaviours worth knowing:

- **Risky tools ask first.** Writing/appending notes, switching Home Assistant
  devices, and opening, closing or focusing an app produce "Should I ...? Say
  yes to confirm." The pending action survives one turn; yes runs it, anything
  else cancels it. Listing running apps never asks.
- **App control by voice.** "Open chrome", "close notepad", "switch to
  spotify", "what apps are open" go through `tools.apps`. Add your own spoken
  names in `tools.apps.aliases` (`browser: chrome`), and use
  `tools.apps.allow` to restrict which apps Jarvis may touch. On Windows it
  uses `start`, so Store apps such as `spotify` work by name; if one is not
  found, put the exact name it has on your machine in `aliases`.
- **Degradation is graceful.** No cloud key, no Ollama, no Piper voice or no
  Chroma embedding model each produce a warning and a reduced mode (local LLM
  only, text instead of speech, facts without conversation recall) instead of a
  crash. `--check` lists all of it.

## Layout

```
config.yaml        single source of truth, fully commented
config.py          loader: env refs, path resolution, validation, --check CLI
requirements.txt   runtime deps; requirements-dev.txt adds pytest
prompts/           system prompt templates (used by brain.py)
audio.py           mic -> wake word -> VAD -> Whisper
brain.py           LLM router + system prompt assembly
memory.py          SQLite user_facts + Chroma conversation recall
memory_cli.py      list/get/set/forget/decay facts from the shell
tools.py           web search, Home Assistant, notes, reminders, shell, apps
tts.py             Piper synthesis and playback
main.py            async entry point
desktop.py         the --app desktop window: server + voice + pywebview/tray/hotkey
ui_server.py       FastAPI dashboard: SSE metrics, WebSocket chat, weather
ui_metrics.py      psutil sampler and alert rules behind the dashboard
ui/                index.html + app.js + style.css (no build step)
doctor.py          preflight checks: audio, models, Ollama, keys
.github/workflows/ci.yml   pytest on 3.11 + full dependency import check
data/              jarvis.db, chroma/, reminders.db   (git-ignored)
models/            whisper/ and piper/ model files    (git-ignored)
logs/ notes/       runtime output and your notes      (git-ignored)
tests/             pytest suite
```

## Configuration

One file, `config.yaml`, and one loader, `config.py`. Everything else imports
`get_config()`; nothing reads YAML or environment variables directly.

```python
from config import get_config
cfg = get_config()
cfg.memory.decay_days           # 90
cfg.routing_order()             # providers to try, in order
cfg.tts_model_path()            # derived from voices_dir + voice
```

Rules the loader enforces:

- **Secrets never live in the file.** `env:OPENAI_API_KEY` (or `${VAR}`) is
  resolved from the environment at load time. A missing variable becomes
  `None` plus a warning, so a local-only setup starts without any cloud keys.
- **Relative paths resolve against the config file**, not the current
  directory, so `python /path/to/main.py` works from anywhere.
- **Unknown keys are errors.** A typo like `assistent:` fails loudly at startup
  instead of silently taking defaults.
- **Cross-field checks.** Wake word requires 16 kHz; shell whitelist entries
  must be bare command names; duplicate providers are rejected; Chroma
  collection names are validated; VAD windows are ordered sensibly.
- **Warnings, not crashes, for the environment.** Missing Piper voice, no
  usable LLM provider, absent Home Assistant token and similar problems are
  reported by `--check` and in the log, and the assistant continues.

`python config.py --check` prints the resolved configuration with secrets
masked (`OPENAI_API_KEY=set`) and lists warnings. `--json` gives the same as
redacted JSON for scripting.

## Memory design

- **Facts** - SQLite table `user_facts(id, key, value, created_at,
  last_accessed, stale)`. Keys are `lowercase_snake_case`, upserted by name.
- **Seeds** - `memory.seed_facts` in config.yaml is written on first run only;
  after that the database owns the values.
- **Recall** - before every LLM call the fact store is queried for the current
  topic and injected as `User facts: name=..., timezone=..., tone=...`, next to
  the relevant past turns from Chroma.
- **Learn** - after each exchange a short, separately budgeted LLM call
  extracts new user-stated facts (preferences, corrections) and upserts them.
- **Decay** - facts untouched for `decay_days` (90) are flagged `stale` and
  surfaced to the model with "may be outdated, confirm if relevant".

## Design decisions

- **All providers over plain HTTP (`httpx`).** One streaming interface yields
  text deltas and a final completion for Ollama, OpenAI and Anthropic, with
  per-provider timeouts, retries for transient failures and a fallback chain
  that only skips ahead when nothing has been spoken yet.
- **SQLite for facts, Chroma for recall.** SQLite is exact, fast and
  inspectable for "what do I know about the user"; Chroma handles fuzzy
  "what did we say about X last week" over embeddings, computed locally.
- **80 ms / 16 kHz audio.** openWakeWord's natural chunk is 1280 samples;
  Whisper wants 16 kHz. Fixing both at the audio layer avoids resampling in
  three modules.
- **Config owns tuning, memory owns facts.** `tts.voice` in config wins for
  synthesis; `preferred_tts_voice` in memory is context for the model, not a
  second source of truth.
- **Safety defaults.** Shell runs only whitelisted bare commands (no shell, no
  absolute or `..` paths, timeout and output cap), Home Assistant is limited to
  `allow_domains`, fact extraction refuses credentials, and
  `confirm_risky_actions: true` gates side effects on a spoken yes.
- **App control is name-only.** `tools.apps` opens, closes, focuses and lists
  desktop apps by bare name: `start` on Windows (knows Store apps and file
  associations), `open -a` on macOS, PATH / `gtk-launch` / `xdg-open` on Linux.
  Names are checked against a plain-name pattern and `tools.apps.aliases`
  targets must be bare names too, so no path, flag, `&` or `;` ever reaches a
  shell. `tools.apps.allow` restricts which apps may be controlled.

## Testing

Each module is importable and testable on its own; heavy dependencies
(microphone, models, Chroma, Piper, HTTP) are injected or faked, so `pytest`
runs headless in about a second. The suite covers config loading and
validation, framing/VAD/wake-word policy, all three streaming adapters with
mock transports, fallback and retry behaviour, fact seeding/decay/recall,
extraction parsing and the sensitive-value guard, every tool, Piper chunking
and WAV decoding, and the conversation/tool/confirmation loop.

The one path tests cannot cover is real audio: PortAudio needs a sound server.
Voice capture and playback are verified by construction and must be tried on
the machine with a microphone.
