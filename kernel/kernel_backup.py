"""
Agent OS — Kernel
Top-level facade. Boot the OS, spawn agents, interact with the system.

Usage:
    async with Kernel.boot() as kernel:
        proc = await kernel.spawn("researcher", task="Summarise X")
        result = await kernel.wait(proc.pid)
        print(result.result)
"""

from __future__ import annotations
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis

from .models import AgentProcess, AgentState, Priority, SchedulerStats
from .process_manager import ProcessManager
from .scheduler import Scheduler, SchedulingPolicy
from .ipc import IPCBus
from .executor import LLMExecutor

logger = logging.getLogger("agent_os.kernel")


class Kernel:
    """
    The Agent OS kernel.

    Wires together:
        ProcessManager  ←→  Scheduler  ←→  LLMExecutor
                                  ↕
                              IPCBus
                                  ↕
                             Redis (persistence)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY_PREEMPTIVE,
        api_key: str | None = None,
    ):
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._pm: ProcessManager | None = None
        self._ipc: IPCBus | None = None
        self._scheduler: Scheduler | None = None
        self._executor: LLMExecutor | None = None
        self._loop_task: asyncio.Task | None = None
        self._scheduling_policy = scheduling_policy
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Boot / shutdown
    # ------------------------------------------------------------------

    async def boot(self):
        logger.info("Agent OS booting...")
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

        self._pm = ProcessManager(self._redis)
        self._ipc = IPCBus(self._redis)
        self._executor = LLMExecutor(self._ipc, api_key=self._api_key)
        self._scheduler = Scheduler(self._pm, policy=self._scheduling_policy)
        self._scheduler.set_executor(self._executor)

        self._loop_task = asyncio.create_task(self._scheduler.run())
        logger.info("Agent OS ready.")
        return self

    async def shutdown(self):
        logger.info("Agent OS shutting down...")
        self._scheduler.stop()
        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._loop_task.cancel()
        await self._pm.flush_all()
        await self._redis.aclose()
        logger.info("Agent OS halted.")

    @classmethod
    @asynccontextmanager
    async def boot_context(cls, **kwargs):
        kernel = cls(**kwargs)
        await kernel.boot()
        try:
            yield kernel
        finally:
            await kernel.shutdown()

    # ------------------------------------------------------------------
    # Public API — mirrors familiar OS syscalls
    # ------------------------------------------------------------------

    async def spawn(
        self,
        name: str,
        task: str,
        *,
        system_prompt: str = "",
        priority: Priority = Priority.NORMAL,
        token_budget: int = 4096,
        tools: list[str] | None = None,
        parent_pid: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> AgentProcess:
        """Spawn a new agent process. Returns immediately; process runs in background."""
        proc = await self._pm.spawn(
            name=name,
            system_prompt=system_prompt,
            task_description=task,
            priority=priority,
            token_budget=token_budget,
            tools=tools,
            parent_pid=parent_pid,
            tags=tags,
        )
        self._ipc.register(proc.pid)
        return proc

    async def wait(self, pid: str, timeout: float = 120.0) -> AgentProcess:
        """Block until the agent finishes. Returns the completed process."""
        return await self._pm.wait(pid, timeout=timeout)

    async def kill(self, pid: str, reason: str = "user request") -> bool:
        return await self._pm.kill(pid, reason)

    def ps(self) -> list[AgentProcess]:
        """List all processes."""
        return self._pm.ps()

    def get(self, pid: str) -> Optional[AgentProcess]:
        return self._pm.get(pid)

    async def send_message(
        self,
        from_pid: str,
        to_pid: str,
        payload,
        channel: str = "default",
    ):
        return await self._ipc.send(from_pid, to_pid, payload, channel=channel)

    async def broadcast(self, from_pid: str, payload, channel: str = "broadcast"):
        return await self._ipc.broadcast(from_pid, payload, channel=channel)

    @property
    def stats(self) -> SchedulerStats:
        return self._scheduler.stats

    @property
    def ipc_stats(self) -> dict:
        return self._ipc.stats
