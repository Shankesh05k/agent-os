"""
Agent OS — LLM Executor
The bridge between the scheduler's abstract "run this agent" command
and actual Anthropic API calls. Handles:
  - Constructing messages from process state
  - Token budget reservation + accounting
  - Tool call dispatch (extensible)
  - Injecting IPC inbox messages as context
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import Any

import anthropic

from .models import AgentProcess
from .ipc import IPCBus
from .scheduler import ExecutorResult

logger = logging.getLogger("agent_os.executor")


class LLMExecutor:
    """
    Executes one scheduling slice for an agent process.
    Called by the Scheduler as: result = await executor(proc)
    """

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS_PER_CALL = 1024      # cap per LLM call (not total budget)

    def __init__(self, ipc_bus: IPCBus, api_key: str | None = None):
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._ipc = ipc_bus

    async def __call__(self, proc: AgentProcess) -> ExecutorResult:
        """Main entry point called by the Scheduler."""

        # 1. Reserve tokens for this call
        requested = min(self.MAX_TOKENS_PER_CALL, proc.budget.available)
        if not proc.budget.reserve(requested):
            return ExecutorResult(tokens_used=0, error="Budget exhausted", done=True)

        # 2. Drain inbox and inject as system context
        inbox_msgs = await self._ipc.receive_all(proc.pid)
        inbox_context = self._format_inbox(inbox_msgs)

        # 3. Build message list
        messages = list(proc.message_history)
        if not messages:
            # First call — inject task
            messages.append({
                "role": "user",
                "content": self._build_initial_prompt(proc, inbox_context),
            })
        elif inbox_context:
            # Subsequent calls — inject any new IPC messages
            messages.append({
                "role": "user",
                "content": f"[SYSTEM: New messages in your inbox]\n{inbox_context}\n\nContinue your task.",
            })
        else:
            messages.append({
                "role": "user",
                "content": "Continue. If your task is complete, say DONE: followed by your final output.",
            })

        # 4. Call the LLM
        try:
            response = await self._client.messages.create(
                model=self.MODEL,
                max_tokens=requested,
                system=self._build_system_prompt(proc),
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error("API error for pid=%s: %s", proc.pid, e)
            return ExecutorResult(tokens_used=0, error=str(e), done=True)

        # 5. Accounting
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        proc.budget.commit(tokens_used)

        # 6. Parse response
        text = response.content[0].text if response.content else ""
        proc.message_history.append({"role": "assistant", "content": text})

        # 7. Check for completion signal
        done = False
        output = None
        if "DONE:" in text:
            done = True
            output = text.split("DONE:", 1)[1].strip()
            logger.info("pid=%s signalled DONE", proc.pid)

        # 8. Check stop reason
        if response.stop_reason == "end_turn" and not done:
            # Agent finished speaking but didn't say DONE — re-queue for next tick
            pass

        return ExecutorResult(
            tokens_used=tokens_used,
            output=output,
            done=done,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, proc: AgentProcess) -> str:
        base = proc.system_prompt or "You are a helpful AI agent."
        kernel_injection = f"""

[AGENT OS KERNEL CONTEXT]
- Your process ID (PID): {proc.pid}
- Your name: {proc.name}
- Token budget remaining: {proc.budget.available} tokens
- Context switches so far: {proc.context_switches}

You are running inside Agent OS. You may be paused and resumed across multiple turns.
When your task is fully complete, respond with:
  DONE: <your final output here>
Otherwise, keep working. If you need to wait for another agent, end your response normally.
"""
        return base + kernel_injection

    def _build_initial_prompt(self, proc: AgentProcess, inbox_context: str) -> str:
        parts = [f"Task: {proc.task_description}"]
        if inbox_context:
            parts.append(f"\nMessages from other agents:\n{inbox_context}")
        return "\n".join(parts)

    def _format_inbox(self, msgs) -> str:
        if not msgs:
            return ""
        lines = []
        for msg in msgs:
            lines.append(
                f"[From PID {msg.sender_pid} | channel={msg.channel}]: {msg.payload}"
            )
        return "\n".join(lines)
