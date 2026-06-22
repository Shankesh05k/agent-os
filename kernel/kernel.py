from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis

from .models import AgentProcess, AgentState, Priority, SchedulerStats
from .process_manager import ProcessManager
from .scheduler import Scheduler, SchedulingPolicy
from .ipc import IPCBus
from .executor import LLMExecutor
from .ollama_executor import OllamaExecutor
from .tool_registry import ToolRegistry, make_default_tools
from .memory_manager import MemoryManager, MemoryAccess

logger = logging.getLogger("agent_os.kernel")


class Kernel:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY_PREEMPTIVE,
        api_key: str | None = None,
        ollama_host: str = "http://172.21.208.1:11434",
        ollama_model: str = "qwen2.5:3b",
        use_ollama: bool = False,
    ):
        self._redis_url = redis_url
        self._redis = None
        self._pm = None
        self._ipc = None
        self._scheduler = None
        self._loop_task = None
        self._scheduling_policy = scheduling_policy
        self._api_key = api_key
        self._ollama_host = ollama_host
        self._ollama_model = ollama_model
        self._use_ollama = use_ollama
        self.tools: ToolRegistry = make_default_tools()
        self.memory: MemoryManager | None = None

    async def boot(self):
        logger.info("Agent OS booting...")
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        self._pm = ProcessManager(self._redis)
        self._ipc = IPCBus(self._redis)
        self.memory = MemoryManager(self._redis)

        if self._use_ollama:
            executor = OllamaExecutor(
                self._ipc, self.tools,
                ollama_host=self._ollama_host,
                model=self._ollama_model,
            )
            logger.info("Using Ollama executor (model=%s)", self._ollama_model)
        else:
            executor = LLMExecutor(self._ipc, api_key=self._api_key)
            logger.info("Using Anthropic executor")

        self._scheduler = Scheduler(self._pm, policy=self._scheduling_policy)
        self._scheduler.set_executor(executor)
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

    async def spawn(self, name, task, *, system_prompt="", priority=Priority.NORMAL,
                    token_budget=4096, tools=None, parent_pid=None, tags=None):
        proc = await self._pm.spawn(
            name=name, system_prompt=system_prompt, task_description=task,
            priority=priority, token_budget=token_budget,
            tools=tools, parent_pid=parent_pid, tags=tags,
        )
        self._ipc.register(proc.pid)
        return proc

    async def wait(self, pid, timeout=120.0):
        return await self._pm.wait(pid, timeout=timeout)

    async def kill(self, pid, reason="user request"):
        return await self._pm.kill(pid, reason)

    def ps(self):
        return self._pm.ps()

    def get(self, pid):
        return self._pm.get(pid)

    async def send_message(self, from_pid, to_pid, payload, channel="default"):
        return await self._ipc.send(from_pid, to_pid, payload, channel=channel)

    async def broadcast(self, from_pid, payload, channel="broadcast"):
        return await self._ipc.broadcast(from_pid, payload, channel=channel)

    @property
    def stats(self):
        return self._scheduler.stats

    @property
    def ipc_stats(self):
        return self._ipc.stats

    @property
    def memory_stats(self):
        return self.memory.stats if self.memory else {}
