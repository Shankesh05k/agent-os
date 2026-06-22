import asyncio
import logging
from unittest.mock import AsyncMock
from kernel.models import AgentProcess, AgentState, Priority
from kernel.process_manager import ProcessManager
from kernel.scheduler import Scheduler, SchedulingPolicy, ExecutorResult
from kernel.ipc import IPCBus

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-25s  %(message)s")

async def main():
    redis_mock = AsyncMock()
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.sadd = AsyncMock(return_value=1)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=1)

    pm = ProcessManager(redis_mock)
    ipc = IPCBus(redis_mock)

    call_counts = {}

    async def mock_executor(proc: AgentProcess) -> ExecutorResult:
        call_counts[proc.pid] = call_counts.get(proc.pid, 0) + 1
        await asyncio.sleep(0.1)
        if call_counts[proc.pid] >= 2:
            return ExecutorResult(tokens_used=50, done=True, output=f"{proc.name} finished!")
        return ExecutorResult(tokens_used=30, done=False)

    scheduler = Scheduler(pm, policy=SchedulingPolicy.PRIORITY_PREEMPTIVE)
    scheduler.set_executor(mock_executor)

    high = await pm.spawn("high-priority-agent", "", "task", priority=Priority.HIGH, token_budget=512)
    normal = await pm.spawn("normal-agent", "", "task", priority=Priority.NORMAL, token_budget=512)
    low = await pm.spawn("low-priority-agent", "", "task", priority=Priority.LOW, token_budget=512)

    for pid in [high.pid, normal.pid, low.pid]:
        ipc.register(pid)

    print(f"\n{'='*55}")
    print("  Agent OS — Mock Demo (no API key needed)")
    print(f"  PIDs: {high.pid} (HIGH)  {normal.pid} (NORMAL)  {low.pid} (LOW)")
    print(f"{'='*55}\n")

    loop_task = asyncio.create_task(scheduler.run())

    await asyncio.gather(
        pm.wait(high.pid, timeout=30),
        pm.wait(normal.pid, timeout=30),
        pm.wait(low.pid, timeout=30),
    )

    scheduler.stop()
    await asyncio.sleep(0.1)
    loop_task.cancel()

    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    for proc in pm.ps():
        print(f"\n  [{proc.name}] ({proc.priority.name})")
        print(f"  Tokens used:      {proc.budget.used}")
        print(f"  Context switches: {proc.context_switches}")
        print(f"  Result:           {proc.result}")

    stats = scheduler.stats
    print(f"\n{'='*55}")
    print(f"  Kernel Stats")
    print(f"  Ticks:            {stats.tick}")
    print(f"  Context switches: {stats.context_switches}")
    print(f"  Total tokens:     {stats.total_tokens_used}")
    print(f"  Uptime:           {stats.uptime:.2f}s")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    asyncio.run(main())
