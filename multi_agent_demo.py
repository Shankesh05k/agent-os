"""
Agent OS — Example: Multi-agent concurrent run

Spawns 3 agents at different priorities and watches them race:
  - HIGH:   Haiku writer (fast, small budget)
  - NORMAL: Fact researcher
  - LOW:    Background summariser

Run with:
    ANTHROPIC_API_KEY=sk-... python -m examples.multi_agent_demo
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kernel import Kernel, Priority, SchedulingPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
)


async def main():
    async with Kernel.boot_context(
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
        scheduling_policy=SchedulingPolicy.PRIORITY_PREEMPTIVE,
    ) as kernel:

        # Spawn 3 agents
        haiku = await kernel.spawn(
            "haiku-writer",
            task="Write a haiku about operating systems scheduling processes. When done, say DONE: followed by the haiku.",
            priority=Priority.HIGH,
            token_budget=512,
        )

        researcher = await kernel.spawn(
            "fact-researcher",
            task=(
                "List 3 interesting facts about how Linux kernel scheduling works. "
                "Keep each fact to one sentence. When done, say DONE: followed by your list."
            ),
            priority=Priority.NORMAL,
            token_budget=1024,
        )

        summariser = await kernel.spawn(
            "bg-summariser",
            task=(
                "Write a one-paragraph explanation of why priority scheduling matters "
                "in operating systems. When done, say DONE: followed by your paragraph."
            ),
            priority=Priority.LOW,
            token_budget=1024,
        )

        print(f"\n{'='*60}")
        print(f"  Agent OS — 3 agents spawned")
        print(f"  PIDs: {haiku.pid} (HIGH)  {researcher.pid} (NORMAL)  {summariser.pid} (LOW)")
        print(f"{'='*60}\n")

        # Print live process table while waiting
        async def monitor():
            while True:
                procs = kernel.ps()
                alive = [p for p in procs if p.state.value not in ("dead",)]
                if not alive:
                    break
                lines = ["  PID      NAME                STATE      TOKENS"]
                for p in procs:
                    lines.append(
                        f"  {p.pid:<8} {p.name:<20} {p.state.value:<10} "
                        f"{p.budget.used}/{p.budget.total_allocated}"
                    )
                print("\n".join(lines))
                print()
                await asyncio.sleep(2)

        monitor_task = asyncio.create_task(monitor())

        # Wait for all three
        results = await asyncio.gather(
            kernel.wait(haiku.pid, timeout=60),
            kernel.wait(researcher.pid, timeout=60),
            kernel.wait(summariser.pid, timeout=60),
        )

        monitor_task.cancel()

        print(f"\n{'='*60}")
        print("  ALL AGENTS DONE")
        print(f"{'='*60}")
        for proc in results:
            print(f"\n[{proc.name}] ({proc.priority.name})")
            print(f"  Tokens: {proc.budget.used}/{proc.budget.total_allocated}")
            print(f"  CPU time: {proc.cpu_time:.2f}s")
            print(f"  Result: {proc.result or proc.error}")

        stats = kernel.stats
        print(f"\n{'='*60}")
        print(f"  Kernel stats")
        print(f"  Ticks:            {stats.tick}")
        print(f"  Context switches: {stats.context_switches}")
        print(f"  Total tokens:     {stats.total_tokens_used}")
        print(f"  LLM calls:        {stats.total_llm_calls}")
        print(f"  Uptime:           {stats.uptime:.1f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
