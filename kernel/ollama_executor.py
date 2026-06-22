"""
Agent OS — Ollama Executor
Talks to a local Ollama instance instead of the Anthropic API.
Supports tool calling via a simple TOOL: protocol parsed from LLM output.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
import aiohttp
from typing import Any

from .models import AgentProcess
from .ipc import IPCBus
from .scheduler import ExecutorResult
from .tool_registry import ToolRegistry

logger = logging.getLogger("agent_os.ollama_executor")


class OllamaExecutor:
    """
    Executes one scheduling slice using a local Ollama model.
    Parses TOOL: calls from LLM output and executes them via ToolRegistry.

    Protocol the LLM uses:
        TOOL: tool_name {"arg": "value"}   ← call a tool
        DONE: final answer here             ← task complete
    """

    def __init__(
        self,
        ipc_bus: IPCBus,
        tool_registry: ToolRegistry,
        ollama_host: str = "http://172.21.208.1:11434",
        model: str = "qwen2.5:3b",
    ):
        self._ipc = ipc_bus
        self._tools = tool_registry
        self._host = ollama_host
        self._model = model

    async def __call__(self, proc: AgentProcess) -> ExecutorResult:
        # Reserve tokens
        requested = min(512, proc.budget.available)
        if not proc.budget.reserve(requested):
            return ExecutorResult(tokens_used=0, error="Budget exhausted", done=True)

        # Drain inbox
        inbox_msgs = await self._ipc.receive_all(proc.pid)
        inbox_context = self._format_inbox(inbox_msgs)

        # Build messages
        messages = list(proc.message_history)
        if not messages:
            messages.append({
                "role": "user",
                "content": self._build_initial_prompt(proc, inbox_context),
            })
        elif inbox_context:
            messages.append({
                "role": "user",
                "content": f"New messages:\n{inbox_context}\n\nContinue your task.",
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Continue. If you need a tool say: TOOL: tool_name {\"arg\": \"value\"}\n"
                    "If done say: DONE: your final answer"
                ),
            })

        # Call Ollama
        try:
            text, tokens_used = await self._call_ollama(messages, proc.system_prompt)
        except Exception as e:
            logger.error("Ollama error for pid=%s: %s", proc.pid, e)
            proc.budget.release_reservation()
            return ExecutorResult(tokens_used=0, error=str(e), done=True)

        proc.budget.commit(tokens_used)
        proc.message_history.append({"role": "assistant", "content": text})
        logger.debug("pid=%s response: %s", proc.pid, text[:120])

        # Check for DONE
        if "DONE:" in text:
            output = text.split("DONE:", 1)[1].strip()
            return ExecutorResult(tokens_used=tokens_used, done=True, output=output)

        # Check for TOOL call
        tool_match = re.search(r'TOOL:\s*(\w+)\s*(\{.*?\})?', text, re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2) or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            result = await self._tools.call(tool_name, args)
            tool_feedback = (
                f"Tool '{tool_name}' result: {result.output}"
                if result.success
                else f"Tool '{tool_name}' failed: {result.error}"
            )
            proc.message_history.append({"role": "user", "content": tool_feedback})
            logger.info("pid=%s used tool %s → %s", proc.pid, tool_name, str(result.output)[:60])
            return ExecutorResult(tokens_used=tokens_used, done=False)

        # Keep going
        return ExecutorResult(tokens_used=tokens_used, done=False)

    async def _call_ollama(self, messages: list[dict], system: str) -> tuple[str, int]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 512, "temperature": 0.7},
        }
        if system:
            payload["system"] = system

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._host}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        text = data["message"]["content"]
        # Ollama returns eval_count for tokens
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        return text, tokens

    def _build_system_prompt(self, proc: AgentProcess) -> str:
        base = proc.system_prompt or "You are a helpful AI agent."
        tools_info = self._tools.format_for_prompt()
        return f"""{base}

{tools_info}

IMPORTANT INSTRUCTIONS:
- To call a tool: TOOL: tool_name {{"arg": "value"}}
- When your task is fully complete: DONE: your final answer
- Be concise. Think step by step.
- PID: {proc.pid} | Tokens left: {proc.budget.available}
"""

    def _build_initial_prompt(self, proc: AgentProcess, inbox_context: str) -> str:
        proc.system_prompt = self._build_system_prompt(proc)
        parts = [f"Task: {proc.task_description}"]
        if inbox_context:
            parts.append(f"\nMessages from other agents:\n{inbox_context}")
        return "\n".join(parts)

    def _format_inbox(self, msgs) -> str:
        if not msgs:
            return ""
        return "\n".join(
            f"[From {m.sender_pid} | {m.channel}]: {m.payload}" for m in msgs
        )
