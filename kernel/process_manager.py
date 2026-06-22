"""
Agent OS — Process Manager
Manages the lifecycle of agent processes: spawn, kill, wait, list.
Redis is the process table. In-memory dict is the hot cache.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Optional

import redis.asyncio as aioredis

from .models import AgentProcess, AgentState, Priority, TokenBudget

logger = logging.getLogger("agent_os.process_manager")


class ProcessManager:
    """
    The kernel's process table.

    Responsibilities:
    - spawn()  — create a new AgentProcess, register in Redis + local cache
    - kill()   — forcibly terminate an agent
    - wait()   — block until a process reaches ZOMBIE state, then collect result
    - ps()     — list all known processes and their states
    - get()    — fetch a single AgentProcess by PID

    Redis key schema:
      agent_os:proc:{pid}        → JSON-serialised AgentProcess snapshot
      agent_os:pids              → Redis Set of all live PIDs
    """

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client
        self._procs: dict[str, AgentProcess] = {}   # hot cache
        self._waiters: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def spawn(
        self,
        name: str,
        system_prompt: str,
        task_description: str,
        *,
        priority: Priority = Priority.NORMAL,
        token_budget: int = 4096,
        tools: list[str] | None = None,
        parent_pid: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> AgentProcess:
        proc = AgentProcess(
            name=name,
            state=AgentState.READY,
            priority=priority,
            budget=TokenBudget(total_allocated=token_budget),
            system_prompt=system_prompt,
            task_description=task_description,
            tools=tools or [],
            parent_pid=parent_pid,
            tags=tags or {},
        )

        if parent_pid and parent_pid in self._procs:
            self._procs[parent_pid].children.append(proc.pid)

        self._procs[proc.pid] = proc
        self._waiters[proc.pid] = asyncio.Event()
        await self._persist(proc)

        logger.info("SPAWN pid=%s name=%r priority=%s budget=%d",
                    proc.pid, name, priority.name, token_budget)
        return proc

    async def kill(self, pid: str, reason: str = "SIGKILL") -> bool:
        proc = self._procs.get(pid)
        if not proc:
            logger.warning("kill: pid=%s not found", pid)
            return False

        if proc.state in (AgentState.DEAD, AgentState.ZOMBIE):
            return True  # already done

        proc.state = AgentState.DEAD
        proc.error = f"Killed: {reason}"
        await self._persist(proc)
        self._signal_waiters(pid)
        logger.info("KILL pid=%s reason=%r", pid, reason)
        return True

    async def wait(self, pid: str, timeout: float = 60.0) -> AgentProcess:
        """Block until the process is ZOMBIE or DEAD, then collect it."""
        if pid not in self._waiters:
            raise KeyError(f"Unknown pid: {pid}")

        try:
            await asyncio.wait_for(self._waiters[pid].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"wait({pid}) timed out after {timeout}s")

        proc = self._procs[pid]
        if proc.state == AgentState.ZOMBIE:
            proc.state = AgentState.DEAD
            await self._persist(proc)
        return proc

    def get(self, pid: str) -> Optional[AgentProcess]:
        return self._procs.get(pid)

    def ps(self) -> list[AgentProcess]:
        """Return all processes sorted by priority then creation time."""
        return sorted(
            self._procs.values(),
            key=lambda p: (p.priority.value, p.created_at),
        )

    def ps_state(self, state: AgentState) -> list[AgentProcess]:
        return [p for p in self._procs.values() if p.state == state]

    # ------------------------------------------------------------------
    # Internal helpers called by the Scheduler
    # ------------------------------------------------------------------

    def mark_running(self, proc: AgentProcess):
        proc.state = AgentState.RUNNING
        proc.last_scheduled_at = time.monotonic()
        proc.context_switches += 1

    def mark_ready(self, proc: AgentProcess):
        proc.state = AgentState.READY

    def mark_blocked(self, proc: AgentProcess):
        proc.state = AgentState.BLOCKED

    def mark_sleeping(self, proc: AgentProcess, wake_at: float):
        proc.state = AgentState.SLEEPING
        proc.tags["wake_at"] = str(wake_at)

    def mark_zombie(self, proc: AgentProcess, result=None, error: str | None = None):
        proc.state = AgentState.ZOMBIE
        proc.result = result
        proc.error = error
        self._signal_waiters(proc.pid)

    def accumulate_cpu_time(self, proc: AgentProcess):
        if proc.last_scheduled_at is not None:
            proc.cpu_time += time.monotonic() - proc.last_scheduled_at
            proc.last_scheduled_at = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(self, proc: AgentProcess):
        """Write a lightweight snapshot to Redis."""
        snapshot = {
            "pid": proc.pid,
            "name": proc.name,
            "state": proc.state.value,
            "priority": proc.priority.name,
            "tokens_used": proc.budget.used,
            "tokens_total": proc.budget.total_allocated,
            "cpu_time": proc.cpu_time,
            "context_switches": proc.context_switches,
            "error": proc.error,
            "tags": proc.tags,
        }
        key = f"agent_os:proc:{proc.pid}"
        await self._redis.set(key, json.dumps(snapshot), ex=3600)
        await self._redis.sadd("agent_os:pids", proc.pid)

    def _signal_waiters(self, pid: str):
        evt = self._waiters.get(pid)
        if evt:
            evt.set()

    # ------------------------------------------------------------------
    # Bulk persist (called periodically by scheduler)
    # ------------------------------------------------------------------

    async def flush_all(self):
        for proc in self._procs.values():
            await self._persist(proc)
