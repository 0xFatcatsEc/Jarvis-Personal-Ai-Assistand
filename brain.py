"""The LLM brain: one streaming interface over Ollama, OpenAI and Anthropic.

`Brain.stream()` yields `TextDelta` events as tokens arrive and exactly one
final `Completion` carrying the full text and any tool calls. `complete()` is
the same call, collected. Providers are tried in `config.routing_order()`;
a provider that fails before emitting anything falls through to the next one
when the failure reason is in `llm.routing.fallback_on`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Literal

import httpx

from config import LlmConfig, ProviderConfig

log = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant", "tool"]
TRANSIENT_REASONS = frozenset({"connection", "timeout", "rate_limit", "server_error"})

DEFAULT_SYSTEM_TEMPLATE = """\
You are {{name}}, a local-first personal assistant running on the user's own machine.
Tone: {{tone}}

User facts:
{{facts}}

{{memories}}Rules:
- Answer in one to three short sentences unless the user asks for detail: your reply is spoken aloud.
- Use the facts above; never invent facts about the user. If you do not know, say so.
- A fact marked "stale" has not been used in a while: confirm it instead of asserting it.
- Call a tool only when it genuinely helps. Ask before anything destructive.
- Never read out internal ids, JSON or raw tool output; summarise it.
{{tools}}
"""


# --------------------------------------------------------------------------- #
# data types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class TextDelta:
    text: str


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


StreamEvent = TextDelta | Completion


class LLMError(RuntimeError):
    """One provider failed. `reason` drives retry and fallback decisions."""

    def __init__(self, reason: str, message: str, provider: str | None = None) -> None:
        self.reason = reason
        self.provider = provider
        super().__init__(f"{provider or 'llm'}: {reason}: {message}")


class AllProvidersFailed(LLMError):
    def __init__(self, failures: list[LLMError]) -> None:
        self.failures = failures
        summary = "; ".join(f"{f.provider}={f.reason}" for f in failures) or "no provider"
        super().__init__("all_failed", summary)


# --------------------------------------------------------------------------- #
# provider adapters
# --------------------------------------------------------------------------- #
async def _ollama_stream(
    provider: ProviderConfig,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    temperature: float,
    client: httpx.AsyncClient,
) -> AsyncIterator[StreamEvent]:
    base = provider.base_url or "http://localhost:11434"
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": [_ollama_message(m) for m in messages],
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens, **provider.options},
    }
    if tools:
        payload["tools"] = [{"type": "function", "function": t} for t in tools]

    text: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    finish: str | None = None
    async with client.stream("POST", f"{base}/api/chat", json=payload, timeout=provider.timeout_s) as response:
        if response.status_code >= 400:
            raise _classify(response.status_code, await response.aread(), provider.name)
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LLMError("bad_response", f"invalid JSON line: {exc}", provider.name) from exc
            if error := chunk.get("error"):
                raise LLMError("bad_response", str(error), provider.name)
            message = chunk.get("message") or {}
            # thinking models (qwen3) stream reasoning in its own field; it is not the answer
            if thought := message.get("thinking"):
                reasoning.append(thought)
            delta = message.get("content") or ""
            if delta:
                text.append(delta)
                yield TextDelta(delta)
            for raw in message.get("tool_calls") or []:
                function = raw.get("function") or {}
                calls.append(
                    ToolCall(
                        id=raw.get("id") or f"ollama-{len(calls)}",
                        name=function.get("name") or "",
                        arguments=_object(function.get("arguments")),
                    )
                )
            if chunk.get("done"):
                finish = chunk.get("done_reason")
    answer = "".join(text)
    if not answer.strip() and reasoning:
        raise LLMError(
            "bad_response",
            f"model returned reasoning but no answer ({len(''.join(reasoning))} chars of thinking); "
            "raise num_predict in the provider options",
            provider.name,
        )
    yield Completion(text=answer, provider=provider.name, model=provider.model, tool_calls=calls, finish_reason=finish)


def _ollama_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "content": message.content, "tool_name": message.name or ""}
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}} for call in message.tool_calls
        ]
    return out


async def _openai_stream(
    provider: ProviderConfig,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    temperature: float,
    client: httpx.AsyncClient,
) -> AsyncIterator[StreamEvent]:
    base = provider.base_url or "https://api.openai.com/v1"
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": [_openai_message(m) for m in messages],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **provider.options,
    }
    if tools:
        payload["tools"] = [{"type": "function", "function": t} for t in tools]
    headers = {"Authorization": f"Bearer {provider.api_key}"}

    text: list[str] = []
    partial: dict[int, dict[str, str]] = {}
    finish: str | None = None
    async with client.stream(
        "POST", f"{base}/chat/completions", json=payload, headers=headers, timeout=provider.timeout_s
    ) as response:
        if response.status_code >= 400:
            raise _classify(response.status_code, await response.aread(), provider.name)
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise LLMError("bad_response", f"invalid SSE data: {exc}", provider.name) from exc
            if chunk.get("error"):
                raise LLMError("bad_response", str(chunk["error"]), provider.name)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if content := delta.get("content"):
                text.append(content)
                yield TextDelta(content)
            for fragment in delta.get("tool_calls") or []:
                slot = partial.setdefault(int(fragment.get("index", 0)), {"id": "", "name": "", "arguments": ""})
                if fragment.get("id"):
                    slot["id"] = fragment["id"]
                function = fragment.get("function") or {}
                if function.get("name"):
                    slot["name"] += function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]
            finish = choice.get("finish_reason") or finish

    calls = [
        ToolCall(id=slot["id"] or f"openai-{index}", name=slot["name"], arguments=_object(slot["arguments"]))
        for index, slot in sorted(partial.items())
    ]
    yield Completion(text="".join(text), provider=provider.name, model=provider.model, tool_calls=calls, finish_reason=finish)


def _openai_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return out


async def _anthropic_stream(
    provider: ProviderConfig,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
    temperature: float,
    client: httpx.AsyncClient,
) -> AsyncIterator[StreamEvent]:
    base = provider.base_url or "https://api.anthropic.com/v1"
    system, conversation = _anthropic_messages(messages)
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": conversation,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        **provider.options,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = [
            {"name": t["name"], "description": t.get("description", ""), "input_schema": t.get("parameters", {})}
            for t in tools
        ]
    headers = {"x-api-key": provider.api_key or "", "anthropic-version": "2023-06-01"}

    text: list[str] = []
    calls: list[ToolCall] = []
    active: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    async with client.stream(
        "POST", f"{base}/messages", json=payload, headers=headers, timeout=provider.timeout_s
    ) as response:
        if response.status_code >= 400:
            raise _classify(response.status_code, await response.aread(), provider.name)
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise LLMError("bad_response", f"invalid SSE data: {exc}", provider.name) from exc
            kind = event.get("type")
            if kind == "error":
                raise LLMError("bad_response", str(event.get("error")), provider.name)
            if kind == "content_block_start":
                index = int(event.get("index", 0))
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    active[index] = {"id": block.get("id", ""), "name": block.get("name", ""), "json": ""}
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    text.append(delta["text"])
                    yield TextDelta(delta["text"])
                elif delta.get("type") == "input_json_delta":
                    slot = active.setdefault(int(event.get("index", 0)), {"id": "", "name": "", "json": ""})
                    slot["json"] += delta.get("partial_json", "")
            elif kind == "content_block_stop":
                slot = active.pop(int(event.get("index", 0)), None)
                if slot is not None:
                    calls.append(
                        ToolCall(id=slot["id"] or f"anthropic-{len(calls)}", name=slot["name"], arguments=_object(slot["json"]))
                    )
            elif kind == "message_delta":
                finish = (event.get("delta") or {}).get("stop_reason") or finish
    yield Completion(text="".join(text), provider=provider.name, model=provider.model, tool_calls=calls, finish_reason=finish)


def _anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system" and m.content)
    conversation: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tools() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            conversation.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            pending_tool_results.append(
                {"type": "tool_result", "tool_use_id": message.tool_call_id or "", "content": message.content}
            )
            continue
        flush_tools()
        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
            conversation.append({"role": "assistant", "content": blocks})
        else:
            conversation.append({"role": message.role, "content": message.content})
    flush_tools()
    return system, conversation


_BACKENDS: dict[str, Callable[..., AsyncIterator[StreamEvent]]] = {
    "ollama": _ollama_stream,
    "openai": _openai_stream,
    "anthropic": _anthropic_stream,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _object(raw: Any) -> dict[str, Any]:
    """Tool arguments as a dict, whether the API sent JSON text or an object."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _classify(status: int, body: bytes, provider: str) -> LLMError:
    detail = body.decode("utf-8", "replace")[:300]
    if status == 429:
        return LLMError("rate_limit", detail, provider)
    if status in (401, 403):
        return LLMError("auth", detail, provider)
    if status >= 500:
        return LLMError("server_error", detail, provider)
    return LLMError("bad_response", f"HTTP {status}: {detail}", provider)


def _http_error(exc: httpx.HTTPError, provider: str) -> LLMError:
    if isinstance(exc, httpx.TimeoutException):
        return LLMError("timeout", str(exc) or "request timed out", provider)
    return LLMError("connection", str(exc) or "connection failed", provider)


# --------------------------------------------------------------------------- #
# brain
# --------------------------------------------------------------------------- #
class Brain:
    """Tries providers in routing order with retries and fallback."""

    def __init__(
        self,
        cfg: LlmConfig,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Any] | None = None,
        backends: dict[str, Callable[..., AsyncIterator[StreamEvent]]] | None = None,
    ) -> None:
        self.cfg = cfg
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep or asyncio.sleep
        self._backends = backends or _BACKENDS

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def providers(self) -> list[ProviderConfig]:
        mode = self.cfg.routing.mode
        if mode == "local_only":
            ordered = [p for p in self.cfg.providers if p.enabled and p.name == "ollama"]
        elif mode == "cloud_only":
            ordered = [p for p in self.cfg.providers if p.enabled and p.name != "ollama"]
        else:
            ordered = [p for p in self.cfg.providers if p.enabled]
        return [p for p in ordered if p.usable]

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        providers = self.providers()
        if not providers:
            raise LLMError(
                "no_provider",
                "no enabled provider is usable (check keys and llm.routing.mode)",
            )
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature
        failures: list[LLMError] = []

        for provider in providers:
            backend = self._backends[provider.name]
            attempt = 0
            while True:
                emitted = False
                completed = False
                try:
                    async for event in backend(
                        provider, messages, tools, max_tokens, temperature, self.client()
                    ):
                        if isinstance(event, TextDelta):
                            emitted = emitted or bool(event.text)
                        else:
                            completed = True
                        yield event
                    return
                except (LLMError, httpx.HTTPError) as raw:
                    exc = raw if isinstance(raw, LLMError) else _http_error(raw, provider.name)
                    if emitted or completed:
                        raise exc
                    if attempt < provider.max_retries and exc.reason in TRANSIENT_REASONS:
                        attempt += 1
                        log.warning("%s: %s; retry %d/%d", provider.name, exc.reason, attempt, provider.max_retries)
                        await self._sleep(0.4 * (2 ** (attempt - 1)))
                        continue
                    if exc.reason not in self.cfg.routing.fallback_on:
                        raise exc
                    log.warning("%s unavailable (%s); trying next provider", provider.name, exc.reason)
                    failures.append(exc)
                    break
        raise AllProvidersFailed(failures)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        final: Completion | None = None
        async for event in self.stream(messages, tools=tools, max_tokens=max_tokens, temperature=temperature):
            if isinstance(event, Completion):
                final = event
        if final is None:
            raise LLMError("bad_response", "provider stream ended without a completion")
        return final


# --------------------------------------------------------------------------- #
# system prompt
# --------------------------------------------------------------------------- #
def load_system_template(prompts_dir: Path) -> str | None:
    path = Path(prompts_dir) / "system.md"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return text or None


def build_system_prompt(
    *,
    name: str = "Jarvis",
    tone: str = "concise, slightly dry humour, like a competent butler",
    facts: str = "",
    memories: Iterable[str] = (),
    tools_enabled: Iterable[str] = (),
    template: str | None = None,
) -> str:
    """Render the system prompt. `{{token}}` placeholders, never str.format."""
    memory_lines = "\n".join(f"- {line}" for line in memories)
    memory_block = f"Relevant past conversation:\n{memory_lines}\n\n" if memory_lines else ""
    tool_names = sorted(tools_enabled)
    tools_block = (
        f"- Tools available: {', '.join(tool_names)}. Use them when they help.\n" if tool_names else ""
    )
    rendered = (template or DEFAULT_SYSTEM_TEMPLATE)
    for token, value in {
        "name": name,
        "tone": tone,
        "facts": facts or "- (nothing yet)",
        "memories": memory_block,
        "tools": tools_block,
    }.items():
        rendered = rendered.replace("{{" + token + "}}", value)
    return rendered.strip()
