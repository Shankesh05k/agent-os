"""
Agent OS — Week 2+3 Demo
Real agents with tools + shared memory, running on local Ollama.

3 agents collaborate:
  - Analyst (HIGH):    uses calculator + get_time tools, writes findings to memory
  - Researcher (NORMAL): reads memory, uses word_count tool
  - Logger (LOW):      appends all activity to shared log in memory

Run:
    python ollama_demo.py
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernel import Kernel, Priority, SchedulingPolicy
from kernel.memory_manager import MemoryAccess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(message)s",
)

OLLAMA_HOST = "http://172.21.208.1:11434"
OLLAMA_MODEL = "qwen2.5:3b"


async def main():
    print(f"\n{'='*60}")
    print("  Agent OS — Week 2+3 Demo")
    print("  Tools + Memory + Ollama (local, free)")
    print(f"{'='*60}\n")

    async with Kernel.boot_context(
        use_ollama=True,
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL,
        scheduling_policy=SchedulingPolicy.PRIORITY_PREEMPTIVE,
    ) as kernel:

        # Add a custom tool that reads from shared memory
        @kernel.tools.register(
            "read_memory",
            "Read a value from shared agent memory by key.",
            {"key": {"description": "The memory key to read"}}
        )
        async def read_memory_tool(key: str) -> str:
            entry = await kernel.memory.read(key, reader_pid="tool")
            if entry:
                return f"Memory[{key}] = {entry.value}"
            return f"Memory[{key}] = (empty)"

        # Spawn 3 agents
        analyst = await kernel.spawn(
            "analyst",
            task=(
                "You are a data analyst. Do these steps one at a time:\n"
                "1. Use get_time tool to get current time\n"
                "2. Use calculator tool to compute 1337 * 42\n"
                "3. When done with both, say DONE: Current time is <time>, calculation result is <result>"
            ),
            priority=Priority.HIGH,
            token_budget=1024,
        )

        researcher = await kernel.spawn(
            "researcher",
            task=(
                "You are a researcher. Do these steps:\n"
                "1. Use word_count tool on this text: 'Agent OS is an operating system for AI agents with scheduling memory and IPC'\n"
                "2. Use calculator tool to multiply that word count by 100\n"
                "3. Say DONE: The text has <N> words, multiplied by 100 is <result>"
            ),
            priority=Priority.NORMAL,
            token_budget=1024,
        )

        logger_agent = await kernel.spawn(
            "logger",
            task=(
                "You are a system logger. Do these steps:\n"
                "1. Use get_time to get the current time\n"
                "2. Use reverse_text tool on the text 'Agent OS running'\n"
                "3. Say DONE: Logged at <time>, reversed text is <result>"
            ),
            priority=Priority.LOW,
            token_budget=1024,
        )

        print(f"  Spawned agents:")
        print(f"  {analyst.pid} — analyst     (HIGH)")
        print(f"  {researcher.pid} — researcher  (NORMAL)")
        print(f"  {logger_agent.pid} — logger      (LOW)")
        print(f"\n  Tools available: {[t.name for t in kernel.tools.list_tools()]}")
        print(f"\n  Waiting for agents to finish...\n")

        # Monitor progress
        async def monitor():
            while True:
                procs = kernel.ps()
                alive = [p for p in procs if p.state.value not in ("dead", "zombie")]
                if not alive:
                    break
                status_line = "  " + "  |  ".join(
                    f"{p.name}:{p.state.value}({p.budget.used}tok)" for p in procs
                )
                print(status_line)
                await asyncio.sleep(3)

        monitor_task = asyncio.create_task(monitor())

        results = await asyncio.gather(
            kernel.wait(analyst.pid, timeout=120),
            kernel.wait(researcher.pid, timeout=120),
            kernel.wait(logger_agent.pid, timeout=120),
            return_exceptions=True,
        )

        monitor_task.cancel()

        print(f"\n{'='*60}")
        print("  RESULTS")
        print(f"{'='*60}")
        for proc in results:
            if isinstance(proc, Exception):
                print(f"\n  ERROR: {proc}")
                continue
            print(f"\n  [{proc.name}] ({proc.priority.name})")
            print(f"  Tokens:  {proc.budget.used}/{proc.budget.total_allocated}")
            print(f"  Switches: {proc.context_switches}")
            if proc.result:
                print(f"  Result:  {proc.result}")
            if proc.error:
                print(f"  Error:   {proc.error}")

        stats = kernel.stats
        print(f"\n{'='*60}")
        print(f"  Kernel Stats")
        print(f"  Ticks:        {stats.tick}")
        print(f"  Switches:     {stats.context_switches}")
        print(f"  Total tokens: {stats.total_tokens_used}")
        print(f"  LLM calls:    {stats.total_llm_calls}")
        print(f"  Uptime:       {stats.uptime:.1f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
