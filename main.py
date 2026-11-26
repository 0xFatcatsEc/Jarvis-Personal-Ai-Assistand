"""Jarvis entry point: wake word -> transcript -> memory -> LLM -> tools -> voice.

    python main.py                 # voice mode (falls back to typed if no mic)
    python main.py --app           # desktop window: dashboard + microphone
    python main.py --app --text    # desktop window, typed chat only
    python main.py --ui            # the same dashboard in a browser tab
    python main.py --text          # typed conversation
    python main.py --once "hello"  # one reply, then exit
    python main.py --check         # validate config and exit

Risky tool calls (writing notes, switching devices) are not executed straight
away. Jarvis asks "Should I ...? Say yes to confirm." and holds the pending
action across one turn; "yes" runs it, anything else cancels.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Awaitable, Callable, Iterable

from brain import (
    Brain,
    Completion,
    LLMError,
    Message,
    TextDelta,
    ToolCall,
    build_system_prompt,
    load_system_template,
)
from config import Config, ConfigError, load_config
from memory import Memory, RecallContext
from tools import ToolError, ToolRegistry, build_registry
from tts import Speaker, TTSError

log = logging.getLogger("jarvis")

MAX_TOOL_ROUNDS = 4
HISTORY_LIMIT = 12
DEFAULT_TONE = "concise, slightly dry humor, like a competent butler"
YES_RE = r"^(yes|yeah|yep|yup|sure|ok|okay|do it|go ahead|please do|confirm)\b"
NO_RE = r"^(no|nope|nah|cancel|never mind|nevermind|don'?t|stop)\b"


class ConfirmationRequired(Exception):
    """A risky tool call is waiting for the user's yes/no."""

    def __init__(self, messages: list[Message], calls: list[ToolCall], index: int, rounds: int) -> None:
        self.messages = messages
        self.calls = calls
        self.index = index
        self.rounds = rounds
        self.user_text = ""
        super().__init__(f"confirm {calls[index].name}")


@dataclass
class _Pending:
    messages: list[Message]
    calls: list[ToolCall]
    index: int
    rounds: int
    user_text: str


class Jarvis:
    def __init__(
        self,
        cfg: Config,
        *,
        brain: Any | None = None,
        memory: Any | None = None,
        tools: Any | None = None,
        speaker: Speaker | None = None,
        listener: Any | None = None,
        write: Callable[[str], None] = print,
        echo: bool = False,
    ) -> None:
        self.cfg = cfg
        self.brain = brain or Brain(cfg.llm)
        self.memory = memory or Memory(cfg.memory)
        self.tools = tools or build_registry(cfg.tools)
        self.speaker = speaker
        self.listener = listener
        self.write = write
        self.echo = echo
        self.speak_replies = False
        self.on_delta: Callable[[str], Awaitable[None]] | None = None
        self.on_voice: Callable[[str, str], Awaitable[None]] | None = None
        self._history: list[Message] = []
        self._pending: _Pending | None = None
        self._tts_warned = False

    @property
    def name(self) -> str:
        return self.cfg.assistant.name

    # ------------------------------------------------------------------ #
    # conversation
    # ------------------------------------------------------------------ #
    async def handle(self, text: str, *, speak: bool | None = None) -> str:
        text = " ".join(text.split())
        if not text:
            return ""
        speak = self.speak_replies if speak is None else speak

        if self._pending is not None:
            pending, self._pending = self._pending, None
            if _matches(text, YES_RE):
                return await self._resume(pending, speak=speak)
            if _matches(text, NO_RE):
                reply = "Okay, cancelled."
                if speak:
                    await self._say(reply)
                return reply
            # Anything else supersedes the pending action: cancel it and carry on.

        try:
            context = await asyncio.to_thread(self.memory.recall_context, text)
            messages = [Message(role="system", content=self.system_prompt(context))]
            messages.extend(self._history)
            messages.append(Message(role="user", content=text))
            reply = await self._converse(messages)
        except ConfirmationRequired as need:
            need.user_text = text
            self._pending = _Pending(need.messages, need.calls, need.index, need.rounds, text)
            reply = self._confirmation_reply(need)
            if speak:
                await self._say(reply)
            return reply
        except LLMError as exc:
            log.error("llm failed: %s", exc)
            reply = "I can't reach a language model right now."
            if speak:
                await self._say(reply)
            return reply
        return await self._finish(text, reply, speak=speak)

    async def _resume(self, pending: _Pending, *, speak: bool) -> str:
        call = pending.calls[pending.index]
        try:
            result = await self._call_tool(call)
            pending.messages.append(
                Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
            )
            await self._execute_calls(pending.messages, pending.calls, pending.index + 1, pending.rounds)
            reply = await self._converse(pending.messages, rounds=max(0, pending.rounds - 1))
        except ConfirmationRequired as need:
            need.user_text = pending.user_text
            self._pending = _Pending(need.messages, need.calls, need.index, need.rounds, pending.user_text)
            reply = self._confirmation_reply(need)
            if speak:
                await self._say(reply)
            return reply
        except LLMError as exc:
            log.error("llm failed after confirmation: %s", exc)
            reply = "I can't reach a language model right now."
            if speak:
                await self._say(reply)
            return reply
        return await self._finish(pending.user_text, reply, speak=speak)

    async def _converse(self, messages: list[Message], *, rounds: int = 0) -> str:
        while rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            completion = await self._ask(messages)
            if not completion.tool_calls:
                return completion.text.strip() or "I don't have an answer for that."
            messages.append(
                Message(role="assistant", content=completion.text, tool_calls=completion.tool_calls)
            )
            await self._execute_calls(messages, completion.tool_calls, 0, rounds)
        return "I stopped after several tool calls without reaching an answer."

    async def _execute_calls(
        self, messages: list[Message], calls: list[ToolCall], index: int, rounds: int
    ) -> None:
        while index < len(calls):
            call = calls[index]
            if self._needs_confirmation(call):
                raise ConfirmationRequired(messages, calls, index, rounds)
            result = await self._call_tool(call)
            messages.append(Message(role="tool", content=result, tool_call_id=call.id, name=call.name))
            index += 1

    def _needs_confirmation(self, call: ToolCall) -> bool:
        return bool(self.cfg.tools.confirm_risky_actions and self.tools.risky(call.name, call.arguments))

    async def _call_tool(self, call: ToolCall) -> str:
        try:
            return await self.tools.call(call.name, call.arguments)
        except ToolError as exc:
            log.info("tool %s refused: %s", call.name, exc)
            return f"Tool error: {exc}"

    async def _ask(self, messages: list[Message]) -> Completion:
        final: Completion | None = None
        async for event in self.brain.stream(messages, tools=self.tools.schemas() or None):
            if isinstance(event, TextDelta):
                if self.echo and event.text:
                    self.write(event.text)
                if self.on_delta is not None and event.text:
                    await self.on_delta(event.text)
            elif isinstance(event, Completion):
                final = event
        if self.echo and final is not None and final.text:
            self.write("\n")
        if final is None:
            raise LLMError("bad_response", "stream ended without a completion")
        return final

    def _confirmation_reply(self, need: ConfirmationRequired) -> str:
        call = need.calls[need.index]
        summary = self.tools.describe(call.name, call.arguments)
        return f"Should I {summary}? Say yes to confirm."

    async def _finish(self, user_text: str, reply: str, *, speak: bool) -> str:
        reply = reply.strip()
        self._history.extend([Message(role="user", content=user_text), Message(role="assistant", content=reply)])
        self._history = self._history[-HISTORY_LIMIT:]
        await asyncio.to_thread(self.memory.remember_turn, user_text, reply)
        await self.memory.learn(user_text, reply, self._extract)
        if speak:
            await self._say(reply)
        return reply

    async def _extract(self, prompt: str) -> str:
        completion = await self.brain.complete(
            [Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=self.cfg.memory.learn.max_tokens,
        )
        return completion.text

    def system_prompt(self, context: RecallContext) -> str:
        values = {fact.key: fact.value for fact in context.facts}
        return build_system_prompt(
            name=values.get("assistant_name") or self.name,
            tone=values.get("tone") or DEFAULT_TONE,
            facts=context.facts_text(),
            memories=context.memory_lines(),
            tools_enabled=self.tools.names(),
            template=load_system_template(self.cfg.assistant.prompts_dir),
        )

    # ------------------------------------------------------------------ #
    # speech
    # ------------------------------------------------------------------ #
    async def _say(self, text: str) -> None:
        if not text.strip() or self.speaker is None:
            return
        if not self.speaker.available:
            if not self._tts_warned:
                log.warning("text-to-speech unavailable; replies are text only")
                self._tts_warned = True
            return
        try:
            await asyncio.to_thread(self.speaker.speak_chunks, text)
        except TTSError as exc:
            if not self._tts_warned:
                log.warning("text-to-speech failed: %s", exc)
                self._tts_warned = True

    # ------------------------------------------------------------------ #
    # loops
    # ------------------------------------------------------------------ #
    async def once(self, text: str) -> str:
        reply = await self.handle(text, speak=self.speak_replies)
        if not self.echo:
            self.write(f"{self.name}: {reply}")
        return reply

    async def run_text(self) -> None:
        self.speak_replies = False
        self.write(f"{self.name} ready. Type your message, or 'quit' to exit.")
        while True:
            try:
                line = await asyncio.to_thread(input, "You: ")
            except EOFError:
                break
            if line.strip().lower() in {"quit", "exit"}:
                break
            if not line.strip():
                continue
            reply = await self.handle(line)
            if not self.echo:
                self.write(f"{self.name}: {reply}")

    async def run_voice(self) -> None:
        from audio import build_listener

        if self.listener is None:
            self.listener = build_listener(self.cfg.audio, self.cfg.stt)
        if self.speaker is None:
            self.speaker = Speaker(self.cfg.tts)
        self.speak_replies = True
        reminders = asyncio.create_task(self._reminder_loop())
        try:
            await self._say(f"{self.name} is ready.")
            async for transcript in self.listener.listen():
                if self.on_voice is not None:
                    await self.on_voice("user", transcript)
                reply = await self.handle(transcript)
                self.write(f"{self.name}: {reply}")
                if self.on_voice is not None:
                    await self.on_voice("assistant", reply)
        finally:
            reminders.cancel()
            try:
                await reminders
            except asyncio.CancelledError:
                pass

    async def _reminder_loop(self) -> None:
        interval = self.cfg.tools.reminders.check_interval_s
        while True:
            try:
                for reminder in self.tools.reminders.due():
                    await self._say(f"Reminder: {reminder.text}")
            except Exception as exc:  # keep speaking even if the db hiccups
                log.warning("reminder check failed: %s", exc)
            await asyncio.sleep(interval)

    async def aclose(self) -> None:
        await self.brain.aclose()
        self.memory.close()
        self.tools.close()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _matches(text: str, pattern: str) -> bool:
    return re.match(pattern, text.strip(), re.IGNORECASE) is not None


def setup_logging(cfg: Config) -> None:
    cfg.assistant.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.assistant.log_level, logging.INFO))
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in (
        logging.StreamHandler(sys.stderr),
        RotatingFileHandler(
            cfg.assistant.log_dir / "jarvis.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
    ):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def check_summary(cfg: Config) -> str:
    providers = ", ".join(p.name for p in cfg.routing_order() if p.usable) or "none"
    lines = [
        f"config OK: {cfg.assistant.name}",
        f"llm       {providers}  [routing: {cfg.llm.routing.mode}]",
        f"voice     {cfg.tts_model_path()}",
        f"tools     {', '.join(cfg.tools.enabled)}",
        f"warnings  {len(cfg.warnings)}",
    ]
    lines.extend(f"  - {warning}" for warning in cfg.warnings)
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis, a local-first voice assistant")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--text", action="store_true", help="typed mode, no microphone")
    parser.add_argument("--ui", action="store_true", help="serve the local web dashboard")
    parser.add_argument(
        "--app",
        action="store_true",
        help="open the dashboard in its own desktop window (add --text for no mic)",
    )
    parser.add_argument("--once", metavar="TEXT", help="answer one message and exit")
    parser.add_argument("--check", action="store_true", help="validate configuration and exit")
    return parser.parse_args(list(argv) if argv is not None else None)


async def _serve(jarvis: Jarvis, args: argparse.Namespace) -> int:
    try:
        if args.once is not None:
            await jarvis.once(args.once)
            return 0
        if args.ui:
            from ui_server import serve_ui

            await serve_ui(jarvis)
            return 0
        if args.text:
            await jarvis.run_text()
            return 0
        try:
            await jarvis.run_voice()
        except Exception as exc:
            from audio import AudioError

            if not isinstance(exc, AudioError):
                raise
            log.error("voice mode unavailable (%s); starting typed mode", exc)
            jarvis.listener = None
            await jarvis.run_text()
        return 0
    finally:
        await jarvis.aclose()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.check:
        print(check_summary(cfg))
        return 0
    cfg.ensure_runtime_dirs()
    setup_logging(cfg)
    jarvis = Jarvis(cfg, echo=False)
    if args.app and args.once is None:
        from desktop import run_app

        try:
            return run_app(cfg, jarvis, voice=not args.text)
        except KeyboardInterrupt:
            print("\nGoodbye.", file=sys.stderr)
            return 130
    try:
        return asyncio.run(_serve(jarvis, args))
    except KeyboardInterrupt:
        print("\nGoodbye.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
