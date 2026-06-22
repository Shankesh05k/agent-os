"""
Agent OS — Tests
Unit tests for the kernel subsystems.
No Redis, no LLM — uses mocks for both.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kernel.models import AgentProcess, AgentState, Priority, TokenBudget, IPCMessage
from kernel.process_manager import ProcessManager
from kernel.scheduler import Scheduler, SchedulingPolicy, ExecutorResult
from kernel.ipc import IPCBus


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_redis_mock():
    r = AsyncMock()
    r.set = AsyncMock(return_value=True)
    r.sadd = AsyncMock(return_value=1)
    r.lpush = AsyncMock(return_value=1)
    r.expire = AsyncMock(return_value=1)
    return r


@pytest.fixture
def redis_mock():
    return make_redis_mock()


@pytest.fixture
def pm(redis_mock):
    return ProcessManager(redis_mock)


@pytest.fixture
def ipc(redis_mock):
    return IPCBus(redis_mock)


# ------------------------------------------------------------------
# TokenBudget
# ------------------------------------------------------------------

class TestTokenBudget:
    def test_initial_available(self):
        b = TokenBudget(total_allocated=1000)
        assert b.available == 1000

    def test_reserve_reduces_available(self):
        b = TokenBudget(total_allocated=1000)
        assert b.reserve(200)
        assert b.available == 800

    def test_reserve_fails_if_insufficient(self):
        b = TokenBudget(total_allocated=100)
        assert not b.reserve(200)

    def test_commit_uses_actual_tokens(self):
        b = TokenBudget(total_allocated=1000)
        b.reserve(500)
        b.commit(300)   # only 300 actually used
        assert b.used == 300
        assert b.reserved == 0
        assert b.available == 700

    def test_utilization(self):
        b = TokenBudget(total_allocated=1000)
        b.reserve(500)
        b.commit(500)
        assert b.utilization == 0.5


# ------------------------------------------------------------------
# ProcessManager
# ------------------------------------------------------------------

class TestProcessManager:
    @pytest.mark.asyncio
    async def test_spawn_creates_process(self, pm):
        proc = await pm.spawn("test-agent", "You are helpful.", "Do X")
        assert proc.pid is not None
        assert proc.state == AgentState.READY
        assert pm.get(proc.pid) is proc

    @pytest.mark.asyncio
    async def test_ps_returns_all(self, pm):
        await pm.spawn("a", "", "task a")
        await pm.spawn("b", "", "task b")
        assert len(pm.ps()) == 2

    @pytest.mark.asyncio
    async def test_kill_sets_dead(self, pm):
        proc = await pm.spawn("agent", "", "task")
        result = await pm.kill(proc.pid, "test")
        assert result is True
        assert proc.state == AgentState.DEAD

    @pytest.mark.asyncio
    async def test_wait_resolves_on_zombie(self, pm):
        proc = await pm.spawn("agent", "", "task")

        async def simulate_completion():
            await asyncio.sleep(0.05)
            pm.mark_zombie(proc, result="hello")

        asyncio.create_task(simulate_completion())
        completed = await pm.wait(proc.pid, timeout=2.0)
        assert completed.result == "hello"

    @pytest.mark.asyncio
    async def test_priority_ordering(self, pm):
        low = await pm.spawn("low", "", "", priority=Priority.LOW)
        high = await pm.spawn("high", "", "", priority=Priority.HIGH)
        normal = await pm.spawn("normal", "", "", priority=Priority.NORMAL)

        ordered = pm.ps()
        names = [p.name for p in ordered]
        assert names.index("high") < names.index("normal")
        assert names.index("normal") < names.index("low")


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

class TestScheduler:
    def make_scheduler(self, pm, policy=SchedulingPolicy.PRIORITY_PREEMPTIVE):
        s = Scheduler(pm, policy=policy)
        return s

    @pytest.mark.asyncio
    async def test_dispatches_ready_process(self, pm):
        proc = await pm.spawn("agent", "", "task")

        call_log = []

        async def mock_executor(p):
            call_log.append(p.pid)
            return ExecutorResult(tokens_used=10, done=True)

        scheduler = self.make_scheduler(pm)
        scheduler.set_executor(mock_executor)
        await scheduler._tick()

        assert proc.pid in call_log
        assert proc.state == AgentState.ZOMBIE

    @pytest.mark.asyncio
    async def test_budget_exhausted_kills_process(self, pm):
        proc = await pm.spawn("agent", "", "task", token_budget=0)

        async def mock_executor(p):
            return ExecutorResult(tokens_used=0, done=False)

        scheduler = self.make_scheduler(pm)
        scheduler.set_executor(mock_executor)
        await scheduler._tick()

        assert proc.state == AgentState.ZOMBIE
        assert "budget" in (proc.error or "").lower()

    @pytest.mark.asyncio
    async def test_priority_preemption_selects_highest(self, pm):
        low = await pm.spawn("low", "", "", priority=Priority.LOW)
        high = await pm.spawn("high", "", "", priority=Priority.HIGH)

        dispatched = []

        async def mock_executor(p):
            dispatched.append(p.name)
            return ExecutorResult(tokens_used=10, done=True)

        scheduler = self.make_scheduler(pm, SchedulingPolicy.PRIORITY_PREEMPTIVE)
        scheduler.set_executor(mock_executor)
        await scheduler._tick()

        assert dispatched[0] == "high"

    @pytest.mark.asyncio
    async def test_round_robin_cycles(self, pm):
        a = await pm.spawn("a", "", "")
        b = await pm.spawn("b", "", "")
        c = await pm.spawn("c", "", "")

        dispatched = []

        async def mock_executor(p):
            dispatched.append(p.name)
            return ExecutorResult(tokens_used=10, done=False)

        scheduler = self.make_scheduler(pm, SchedulingPolicy.ROUND_ROBIN)
        scheduler.set_executor(mock_executor)

        # 3 ticks should hit all 3 agents
        for _ in range(3):
            await scheduler._tick()

        assert set(dispatched) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_accumulates_stats(self, pm):
        await pm.spawn("a", "", "")

        async def mock_executor(p):
            return ExecutorResult(tokens_used=42, done=True)

        scheduler = self.make_scheduler(pm)
        scheduler.set_executor(mock_executor)
        await scheduler._tick()

        assert scheduler.stats.total_tokens_used == 42
        assert scheduler.stats.total_llm_calls == 1
        assert scheduler.stats.context_switches == 1


# ------------------------------------------------------------------
# IPC Bus
# ------------------------------------------------------------------

class TestIPCBus:
    @pytest.mark.asyncio
    async def test_send_receive(self, ipc):
        ipc.register("proc-a")
        ipc.register("proc-b")

        await ipc.send("proc-a", "proc-b", "hello", channel="greet")
        msg = await ipc.receive("proc-b", timeout=0.1)

        assert msg is not None
        assert msg.payload == "hello"
        assert msg.channel == "greet"
        assert msg.sender_pid == "proc-a"

    @pytest.mark.asyncio
    async def test_broadcast_reaches_all(self, ipc):
        for pid in ["a", "b", "c", "sender"]:
            ipc.register(pid)

        count = await ipc.broadcast("sender", "ping")
        assert count == 3

        for pid in ["a", "b", "c"]:
            msg = await ipc.receive(pid, timeout=0.1)
            assert msg is not None
            assert msg.payload == "ping"

    @pytest.mark.asyncio
    async def test_unregistered_drops_message(self, ipc):
        ipc.register("sender")
        await ipc.send("sender", "ghost", "oops")
        assert ipc.stats["messages_dropped"] == 1

    @pytest.mark.asyncio
    async def test_receive_all_drains_queue(self, ipc):
        ipc.register("inbox")
        for i in range(5):
            await ipc.send("other", "inbox", f"msg-{i}")
        msgs = await ipc.receive_all("inbox")
        assert len(msgs) == 5

    @pytest.mark.asyncio
    async def test_reply_correlates_message(self, ipc):
        ipc.register("alice")
        ipc.register("bob")

        original = await ipc.send("alice", "bob", "question?", channel="qa")
        await ipc.reply(original, "bob", "answer!")

        reply = await ipc.receive("alice", timeout=0.1)
        assert reply.reply_to == original.msg_id
        assert reply.payload == "answer!"
