/* Jarvis dashboard - vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const RING_RADIUS = 208;
  const RING_LENGTH = 2 * Math.PI * RING_RADIUS;

  const app = {
    meta: null,
    thresholds: { cpu_warn: 80, cpu_critical: 90, ram_warn: 85, gpu_warn: 90, disk_warn: 90 },
    streamEl: null,
    busy: false,
    ws: null,
    retries: 0,
    cores: 0,
    terminalAutoScroll: true,
  };

  /* ---------------------------------------------------------------- utils */
  const clamp = (v) => Math.max(0, Math.min(100, Number(v) || 0));
  const setText = (id, text) => {
    const el = $(id);
    if (el) el.textContent = text;
  };
  const fmtBytes = (n) => {
    if (n === null || n === undefined) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = Number(n);
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    const digits = value >= 100 || unit === 0 ? 0 : 1;
    return `${value.toFixed(digits)} ${units[unit]}`;
  };
  const fmtRate = (bps) => (bps === null || bps === undefined ? "—" : `${fmtBytes(bps)}/s`);
  const fmtPct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v)}%`);
  const fmtDuration = (seconds) => {
    if (!seconds && seconds !== 0) return "—";
    const total = Math.round(seconds);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };
  const fmtClock = (date) =>
    date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });

  const levelClass = (level) => {
    const upper = String(level || "").toUpperCase();
    if (upper === "ERROR" || upper === "CRITICAL") return "error";
    if (upper === "WARNING" || upper === "WARN") return "warn";
    return "info";
  };

  /* ---------------------------------------------------------------- clock */
  function tickClock() {
    const now = new Date();
    setText("clock", fmtClock(now));
    setText(
      "date",
      now.toLocaleDateString([], { weekday: "short", day: "2-digit", month: "short", year: "numeric" })
    );
  }

  /* ------------------------------------------------------------- connection */
  function setLink(kind, label) {
    const dot = $("conn-dot");
    dot.className = `dot ${kind}`;
    setText("conn-label", label);
  }

  function setCoreState(kind, detail) {
    const core = $("core");
    core.classList.toggle("thinking", kind === "thinking");
    core.classList.toggle("error", kind === "error");
    const labels = { idle: "Standing by", thinking: "Processing", error: "Fault detected" };
    setText("core-state", labels[kind] || kind);
    if (detail) setText("core-sub", detail);
    const tag = $("core-tag");
    tag.textContent = kind.toUpperCase();
    tag.classList.toggle("busy", kind === "thinking");
    tag.classList.toggle("fault", kind === "error");
  }

  /* ------------------------------------------------------------------ render */
  function bar(id, value) {
    const fill = $(id);
    if (!fill) return;
    const v = clamp(value);
    fill.style.width = `${v}%`;
    fill.classList.toggle("warn", v >= app.thresholds.cpu_warn && v < app.thresholds.cpu_critical);
    fill.classList.toggle("critical", v >= app.thresholds.cpu_critical);
  }

  function renderCores(perCore, count) {
    const container = $("sys-cores");
    if (!container) return;
    const n = Math.min(count || perCore.length || 0, 64);
    if (n !== app.cores) {
      container.innerHTML = "";
      for (let i = 0; i < n; i += 1) {
        const barEl = document.createElement("div");
        barEl.className = "core-bar";
        barEl.appendChild(document.createElement("i"));
        container.appendChild(barEl);
      }
      app.cores = n;
    }
    Array.from(container.children).forEach((barEl, index) => {
      const fill = barEl.firstElementChild;
      const value = clamp(perCore[index] ?? 0);
      fill.style.height = `${Math.max(3, value)}%`;
      fill.classList.toggle("warn", value >= 80 && value < 95);
      fill.classList.toggle("critical", value >= 95);
    });
  }

  function renderRing(load) {
    const progress = $("core-progress");
    if (!progress) return;
    const filled = (RING_LENGTH * clamp(load)) / 100;
    progress.style.strokeDasharray = `${filled} ${RING_LENGTH}`;
    setText("core-load-value", Math.round(clamp(load)));
  }

  function renderSystem(system) {
    if (!system) return;
    const cpu = system.cpu || {};
    const mem = system.memory || {};
    const disk = system.disk || {};
    const net = system.net || {};
    const host = system.host || {};

    setText("sys-cpu-value", fmtPct(cpu.percent));
    bar("sys-cpu-bar", cpu.percent);
    renderCores(cpu.per_core || [], cpu.cores);
    renderRing(cpu.percent);

    setText("sys-ram-value", fmtPct(mem.percent));
    bar("sys-ram-bar", mem.percent);
    setText("sys-ram-detail", `${fmtBytes(mem.used)} / ${fmtBytes(mem.total)} · ${fmtBytes(mem.available)} free`);

    setText("sys-disk-value", fmtPct(disk.percent));
    bar("sys-disk-bar", disk.percent);
    setText("sys-disk-detail", `${fmtBytes(disk.used)} / ${fmtBytes(disk.total)} used`);

    setText("sys-net-up", fmtRate(net.up_bps));
    setText("sys-net-down", fmtRate(net.down_bps));

    const load = Array.isArray(cpu.load) ? ` · load ${cpu.load.map((v) => Number(v).toFixed(2)).join(" ")}` : "";
    setText("sys-hostline", `${host.name || "host"} · ${host.platform || ""} · up ${fmtDuration(host.uptime_s)}${load}`);
  }

  function renderGpu(gpu) {
    const tag = $("gpu-tag");
    if (!gpu) {
      tag.textContent = "CPU";
      tag.classList.remove("live");
      setText("gpu-name", "No NVIDIA GPU detected");
      setText("gpu-vram-value", "—");
      setText("gpu-util", "—");
      setText("gpu-temp", "—");
      $("gpu-vram-bar").style.width = "0%";
      setText("gpu-note", "Inference runs on the CPU. Install an NVIDIA driver to see VRAM here.");
      return;
    }
    tag.textContent = "NVIDIA";
    tag.classList.add("live");
    setText("gpu-name", gpu.name || "GPU");
    const used = gpu.vram_used_mb;
    const total = gpu.vram_total_mb;
    setText("gpu-vram-value", used != null && total != null ? `${(used / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GB` : "—");
    const vramBar = $("gpu-vram-bar");
    vramBar.style.width = `${clamp(gpu.vram_percent)}%`;
    vramBar.classList.toggle("critical", (gpu.vram_percent || 0) >= app.thresholds.gpu_warn);
    setText("gpu-util", fmtPct(gpu.util_percent));
    setText("gpu-temp", gpu.temperature_c != null ? `${Math.round(gpu.temperature_c)}°C` : "—");
    setText("gpu-note", gpu.vram_percent != null ? `${fmtPct(gpu.vram_percent)} of VRAM allocated` : "VRAM usage unavailable");
  }

  function renderWeather(weather) {
    if (!weather) {
      setText("wx-temp", "--°");
      setText("wx-cond", "Weather unavailable");
      setText("wx-feels", "No route to Open-Meteo");
      setText("wx-city", "OFFLINE");
      setText("wx-wind", "—");
      setText("wx-humidity", "—");
      setText("tw-temp", "--°");
      setText("tw-text", "weather · —");
      setText("loc-name", "Location unknown");
      setText("loc-coords", "—");
      return;
    }
    const temp = weather.temperature_c != null ? `${Math.round(weather.temperature_c)}°C` : "--°";
    setText("wx-temp", temp);
    setText("wx-cond", weather.condition || "—");
    setText("wx-feels", `Feels like ${weather.apparent_c != null ? `${Math.round(weather.apparent_c)}°C` : "—"}`);
    setText("wx-city", weather.name || "—");
    setText("wx-wind", weather.wind_kph != null ? `${Math.round(weather.wind_kph)} km/h` : "—");
    setText("wx-humidity", weather.humidity != null ? `${Math.round(weather.humidity)}%` : "—");
    setText("tw-temp", temp);
    setText("tw-text", `${weather.condition || ""} · ${weather.name || ""}`.trim());
    setText("loc-name", weather.name || "—");
    const lat = weather.latitude;
    const lon = weather.longitude;
    setText(
      "loc-coords",
      lat != null && lon != null
        ? `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"} · ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}`
        : "—"
    );
    const pinned = app.meta && app.meta.location && app.meta.location.latitude != null;
    setText("loc-source", `Source: ${pinned ? "config ui.location" : "IP geolocation"}`);
  }

  function renderActivity(activity) {
    const list = $("activity-list");
    if (!activity || activity.length === 0) {
      list.innerHTML = '<li class="activity-empty">No events yet.</li>';
      return;
    }
    const fragment = document.createDocumentFragment();
    activity.slice(-8).forEach((entry) => {
      const item = document.createElement("li");
      item.className = levelClass(entry.level);
      const time = document.createElement("span");
      time.className = "at";
      time.textContent = entry.time || "";
      const msg = document.createElement("span");
      msg.className = "msg";
      msg.textContent = entry.message || "";
      msg.title = entry.raw || entry.message || "";
      item.append(time, msg);
      fragment.appendChild(item);
    });
    list.replaceChildren(fragment);
  }

  function renderTerminal(activity) {
    const pre = $("terminal-list");
    if (!activity || activity.length === 0) {
      pre.textContent = "logs/jarvis.log\n(no entries yet)";
      return;
    }
    const nearBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 12;
    pre.textContent = activity.map((entry) => entry.raw || entry.message).slice(-12).join("\n");
    if (nearBottom) pre.scrollTop = pre.scrollHeight;
  }

  function renderCore(core) {
    if (!core) return;
    setText("core-name", core.name || "JARVIS");
    setCoreState(core.state || "idle", core.detail);
    setText("core-model", core.model || "—");
    setText("core-latency", core.last_latency_ms != null ? `${(core.last_latency_ms / 1000).toFixed(1)}s` : "—");
    setText("core-uptime", fmtDuration(core.uptime_s));
    setText("core-turns", String(core.turns || 0));
    setText("st-provider", core.provider || "—");
    setText("st-routing", core.routing || "—");
    setText("st-context", core.context ? `${core.context} tokens` : "—");
  }

  function renderAlerts(alerts) {
    const badge = $("alert-badge");
    const list = $("alert-list");
    const items = alerts || [];
    const critical = items.find((a) => a.severity === "critical");
    badge.hidden = !critical;
    if (critical) badge.textContent = critical.label;
    list.replaceChildren();
    items.forEach((alert) => {
      const chip = document.createElement("span");
      chip.className = `alert-chip ${alert.severity}`;
      chip.textContent = `${alert.label} · ${alert.detail}`;
      list.appendChild(chip);
    });
  }

  function renderHost(host) {
    if (!host) return;
    setText("st-host", `${host.name || "—"} · ${host.platform || "—"}`);
    setText("st-python", host.python || "—");
    setText("st-uptime", fmtDuration(host.uptime_s));
  }

  function render(payload) {
    renderSystem(payload.system);
    renderGpu((payload.system || {}).gpu);
    renderAlerts((payload.system || {}).alerts);
    renderHost((payload.system || {}).host);
    renderWeather(payload.weather);
    renderActivity(payload.activity);
    renderTerminal(payload.activity);
    renderCore(payload.core);
  }

  /* ------------------------------------------------------------------- chat */
  function scrollChat() {
    const log = $("chat-log");
    log.scrollTop = log.scrollHeight;
  }

  function addMessage(role, text, { error = false, meta = "" } = {}) {
    const hint = document.querySelector(".chat-hint");
    if (hint) hint.remove();
    const bubble = document.createElement("div");
    bubble.className = `msg ${role === "user" ? "user" : "bot"}${error ? " error" : ""}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = role === "user" ? "YOU" : error ? "SYSTEM" : "JARVIS";
    const body = document.createElement("span");
    body.className = "body";
    body.textContent = text;
    bubble.append(who, body);
    if (meta) {
      const note = document.createElement("span");
      note.className = "meta";
      note.textContent = meta;
      bubble.appendChild(note);
    }
    $("chat-log").appendChild(bubble);
    scrollChat();
    return bubble;
  }

  function setChatStatus(label, online) {
    const tag = $("chat-status");
    tag.textContent = label;
    tag.classList.toggle("live", Boolean(online));
  }

  function setBusy(busy) {
    app.busy = busy;
    $("chat-form").querySelector("button").disabled = busy;
  }

  function handleEvent(event) {
    if (!event || !event.type) return;
    if (event.type === "status") {
      setCoreState(event.state || "thinking", event.text);
      if (!app.streamEl) {
        app.streamEl = addMessage("bot", "");
        app.streamEl.classList.add("streaming");
      }
      return;
    }
    if (event.type === "delta") {
      if (!app.streamEl) {
        app.streamEl = addMessage("bot", "");
        app.streamEl.classList.add("streaming");
      }
      const body = app.streamEl.querySelector(".body");
      body.textContent += event.text || "";
      scrollChat();
      return;
    }
    if (event.type === "turn") {
      // A turn that happened outside this page (spoken input in the desktop
      // window) - show it here so one conversation lists everything.
      addMessage(event.role === "user" ? "user" : "bot", event.text || "");
      setCoreState(event.role === "user" ? "thinking" : "idle",
        event.role === "user" ? "Voice input" : "Standing by");
      scrollChat();
      return;
    }
    if (event.type === "reply") {
      const bubble = app.streamEl || addMessage("bot", "");
      bubble.classList.remove("streaming");
      bubble.querySelector(".body").textContent = event.text || "(empty reply)";
      if (event.latency_ms != null) {
        const note = document.createElement("span");
        note.className = "meta";
        note.textContent = `${(event.latency_ms / 1000).toFixed(1)}s`;
        bubble.appendChild(note);
      }
      app.streamEl = null;
      setBusy(false);
      setCoreState("idle", "Standing by");
      scrollChat();
      return;
    }
    if (event.type === "error") {
      const bubble = app.streamEl || addMessage("bot", "", { error: true });
      bubble.classList.remove("streaming");
      bubble.classList.add("error");
      bubble.querySelector(".who").textContent = "SYSTEM";
      bubble.querySelector(".body").textContent = event.text || "Something went wrong.";
      app.streamEl = null;
      setBusy(false);
      setCoreState("error", "See conversation");
      scrollChat();
    }
  }

  function connectSocket() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    let socket;
    try {
      socket = new WebSocket(`${proto}://${location.host}/api/chat`);
    } catch (err) {
      scheduleSocketRetry();
      return;
    }
    app.ws = socket;
    socket.onopen = () => {
      app.retries = 0;
      setChatStatus("ONLINE", true);
    };
    socket.onmessage = (event) => {
      try {
        handleEvent(JSON.parse(event.data));
      } catch (err) {
        /* ignore malformed frame */
      }
    };
    socket.onclose = () => {
      app.ws = null;
      setChatStatus("RECONNECTING", false);
      if (app.busy) {
        setBusy(false);
        setCoreState("error", "Link dropped");
      }
      scheduleSocketRetry();
    };
    socket.onerror = () => socket.close();
  }

  function scheduleSocketRetry() {
    app.retries += 1;
    const delay = Math.min(15000, 800 * 2 ** Math.min(app.retries, 4));
    setTimeout(connectSocket, delay);
  }

  function sendMessage(text) {
    if (!text || app.busy) return;
    if (!app.ws || app.ws.readyState !== WebSocket.OPEN) {
      addMessage("bot", "No link to the Jarvis core. Retrying — try again in a moment.", { error: true });
      return;
    }
    addMessage("user", text);
    setBusy(true);
    setCoreState("thinking", "Processing your request");
    app.ws.send(JSON.stringify({ text }));
  }

  /* --------------------------------------------------------------- bootstrap */
  async function loadState() {
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      if (!response.ok) return;
      const data = await response.json();
      app.meta = data;
      if (data.thresholds) Object.assign(app.thresholds, data.thresholds);
      renderCore(data.core);
      (data.turns || []).forEach((turn) => {
        addMessage(turn.role === "user" ? "user" : "bot", turn.text);
      });
    } catch (err) {
      /* the SSE stream will fill the panels anyway */
    }
  }

  function connectMetrics() {
    const source = new EventSource("/api/metrics");
    source.onopen = () => setLink("online", "LINKED");
    source.onmessage = (event) => {
      try {
        render(JSON.parse(event.data));
      } catch (err) {
        /* skip a malformed tick */
      }
    };
    source.onerror = () => setLink("offline", "LINK DOWN");
  }

  function wireDock() {
    document.querySelectorAll(".dock button").forEach((button) => {
      button.addEventListener("click", () => {
        const target = $(button.dataset.panel);
        if (!target) return;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        document.querySelectorAll(".dock button").forEach((other) => other.classList.remove("active"));
        button.classList.add("active");
        target.animate?.(
          [
            { boxShadow: "0 0 0 1px rgba(34,227,255,0.9)" },
            { boxShadow: "0 0 0 1px rgba(34,227,255,0.2)" },
          ],
          { duration: 900, easing: "ease-out" }
        );
      });
    });
  }

  function init() {
    tickClock();
    setInterval(tickClock, 1000);
    wireDock();

    $("chat-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const input = $("chat-input");
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      sendMessage(text);
      input.focus();
    });

    loadState();
    connectMetrics();
    connectSocket();

    if (window.matchMedia("(min-width: 901px)").matches) {
      $("chat-input").focus({ preventScroll: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
