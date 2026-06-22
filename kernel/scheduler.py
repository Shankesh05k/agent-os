"""
Agent OS — Scheduler
Decides which agents run when, enforces token budgets, and drives the
main kernel loop. Two policies implemented: round-robin and priority preemption.
"""

from __future__ import annotations
import asyncio
import logging
import time
from enum import Enum
from typing import Optional

from .models import AgentProcess, AgentState, Priority, SchedulerStats
from .process_manager import ProcessManager

logger = logging.getLogger("agent_os.scheduler")


class SchedulingPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    PRIORITY_PREEMPTIVE = "priority_preemptive"


class Scheduler:
    """
    The kernel scheduler.

    Each "tick" the scheduler:
      1. Moves sleeping agents that have passed their wake time → READY
      2. Picks the next READY agent according to the active policy
      3. Grants it a time slice (calls its executor coroutine)
      4. After the slice, updates process state + token accounting
      5. Persists stats

    The executor is injected — the scheduler doesn't call the LLM directly.
    That keeps it testable: pass a mock executor in tests.

    Executor signature:
        async def executor(proc: AgentProcess) -> ExecutorResult
    """

    TICK_INTERVAL = 0.05        # seconds between scheduler ticks
    DEFAULT_TIME_SLICE = 30.0   # max seconds per scheduling slot

    def __init__(
        self,
        process_manager: ProcessManager,
        policy: SchedulingPolicy = SchedulingPolicy.PRIORITY_PREEMPTIVE,
    ):
        self._pm = process_manager
        self.policy = policy
        self.stats = SchedulerStats()
        self._running = False
        self._executor = None
        self._rr_cursor: int = 0        # round-robin position

    # ------------------------------------------------------------------
    # Boot / shutdown
    # ------------------------------------------------------------------

    def set_executor(self, executor):
        """Inject the agent executor coroutine."""
        self._executor = executor

    async def run(self):
        """Main kernel loop. Call once; it runs until stop() is called."""
        if self._executor is None:
            raise RuntimeError("No executor set — call set_executor() first.")
        self._running = True
        logger.info("Scheduler started (policy=%s)", self.policy.value)
        while self._running:
            await self._tick()
            await asyncio.sleep(self.TICK_INTERVAL)
        logger.info("Scheduler stopped. Stats: ticks=%d switches=%d tokens=%d",
                    self.stats.tick, self.stats.context_switches, self.stats.total_tokens_used)

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    async def _tick(self):
        self.stats.tick += 1
        self._wake_sleepers()

        ready = self._pm.ps_state(AgentState.READY)
        if not ready:
            return

        proc = self._select(ready)
        if proc is None:
            return

        await self._dispatch(proc)

    def _wake_sleepers(self):
        now = time.monotonic()
        for proc in self._pm.ps_state(AgentState.SLEEPING):
            wake_at = float(proc.tags.get("wake_at", "0"))
            if now >= wake_at:
                self._pm.mark_ready(proc)
                logger.debug("WAKE pid=%s", proc.pid)

    def _select(self, ready: list[AgentProcess]) -> Optional[AgentProcess]:
        if self.policy == SchedulingPolicy.ROUND_ROBIN:
            return self._select_rr(ready)
        return self._select_priority(ready)

    def _select_rr(self, ready: list[AgentProcess]) -> AgentProcess:
        # Cycle through ready processes in order of creation time
        ordered = sorted(ready, key=lambda p: p.created_at)
        self._rr_cursor = self._rr_cursor % len(ordered)
        chosen = ordered[self._rr_cursor]
        self._rr_cursor += 1
        return chosen

    def _select_priority(self, ready: list[AgentProcess]) -> AgentProcess:
        """
        Strict priority preemption. Within the same priority, use
        arrival-time ordering (oldest first) to prevent starvation.
        """
        return min(ready, key=lambda p: (p.priority.value, p.created_at))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, proc: AgentProcess):
        # Budget exhausted → kill before wasting a slot
        if proc.budget.available <= 0:
            logger.warning("BUDGET_EXHAUSTED pid=%s name=%r", proc.pid, proc.name)
            self._pm.mark_zombie(proc, error="Token budget exhausted")
            self.stats.total_agents_dead += 1
            return

        self._pm.mark_running(proc)
        self.stats.context_switches += 1
        logger.debug("DISPATCH pid=%s name=%r tick=%d", proc.pid, proc.name, self.stats.tick)

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._executor(proc),
                timeout=self.DEFAULT_TIME_SLICE,
            )
            elapsed = time.monotonic() - start

            self._pm.accumulate_cpu_time(proc)
            self.stats.total_tokens_used += result.tokens_used
            self.stats.total_llm_calls += 1
            proc.budget.commit(result.tokens_used)

            if result.done:
                self._pm.mark_zombie(proc, result=result.output)
                self.stats.total_agents_dead += 1
                logger.info("DONE pid=%s name=%r tokens=%d elapsed=%.2fs",
                            proc.pid, proc.name, result.tokens_used, elapsed)
            elif result.blocked:
                self._pm.mark_blocked(proc)
            elif result.sleep_for:
                wake_at = time.monotonic() + result.sleep_for
                self._pm.mark_sleeping(proc, wake_at)
            else:
                self._pm.mark_ready(proc)

        except asyncio.TimeoutError:
            self._pm.accumulate_cpu_time(proc)
            self._pm.mark_ready(proc)
            logger.warning("TIMEOUT pid=%s name=%r — re-queued", proc.pid, proc.name)

        except Exception as exc:
            self._pm.accumulate_cpu_time(proc)
            self._pm.mark_zombie(proc, error=str(exc))
            self.stats.total_agents_dead += 1
            logger.exception("CRASH pid=%s name=%r: %s", proc.pid, proc.name, exc)

        finally:
            proc.budget.release_reservation()


# ------------------------------------------------------------------
# Executor result contract
# ------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutorResult:
    """
    What an executor must return after each scheduling slice.
    The scheduler uses these fields to decide the agent's next state.
    """
    tokens_used: int = 0
    output: Any = None
    done: bool = False        # Agent has completed its task
    blocked: bool = False     # Agent is waiting on an external event
    sleep_for: float = 0.0    # Seconds to sleep before re-queuing
    error: str | None = None
