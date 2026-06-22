"""
Agent OS — Week 5: Grand Finale Demo
Multi-agent research pipeline with live web dashboard.

Pipeline:
  1. Planner   (HIGH)    — breaks the topic into research questions
  2. Researcher (NORMAL) — investigates each question using tools
  3. Writer    (NORMAL)  — synthesises findings into a report
  4. Critic    (LOW)     — reviews and scores the report

Open http://localhost:8000 in your browser BEFORE running to watch live.

Run:
    python grand_finale.py
"""

import asyncio
import logging
import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernel import Kernel, Priority, SchedulingPolicy
from kernel.memory_manager import MemoryAccess
from kernel.dashboard import DashboardServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(message)s",
)

OLLAMA_HOST = "http://172.21.208.1:11434"
OLLAMA_MODEL = "qwen2.5:3b"
TOPIC = "the advantages and disadvantages of microkernel vs monolithic operating system architectures"


async def run_pipeline(kernel: Kernel, dashboard: DashboardServer):
    # ----------------------------------------------------------------
    # Add pipeline-specific tools
    # ----------------------------------------------------------------

    @kernel.tools.register(
        "save_to_memory",
        "Save a finding or result to shared memory so other agents can read it.",
        {
            "key": {"description": "Memory key, e.g. 'research:findings'"},
            "value": {"description": "The content to save"},
        }
    )
    async def save_to_memory(key: str, value: str) -> str:
        await kernel.memory.write(key, value, owner_pid="tool", access=MemoryAccess.PUBLIC)
        return f"Saved to memory[{key}]"

    @kernel.tools.register(
        "read_memory",
        "Read a value from shared memory by key.",
        {"key": {"description": "The memory key to read"}}
    )
    async def read_memory_tool(key: str) -> str:
        entry = await kernel.memory.read(key, reader_pid="tool")
        if entry:
            return str(entry.value)
        return f"(nothing stored at '{key}' yet)"

    @kernel.tools.register(
        "list_memory_keys",
        "List all keys currently stored in shared memory.",
        {}
    )
    async def list_memory_keys() -> str:
        keys = kernel.memory.list_keys("tool")
        return ", ".join(keys) if keys else "(memory is empty)"

    # ----------------------------------------------------------------
    # Spawn the pipeline agents
    # ----------------------------------------------------------------

    dashboard.log_event("INFO", f"Starting pipeline: {TOPIC[:40]}...")

    planner = await kernel.spawn(
        "planner",
        task=(
            f"You are a research planner. Your topic is: '{TOPIC}'\n\n"
            "Do this in order:\n"
            "1. Use save_to_memory tool with key='plan:topic' and value=the topic\n"
            "2. Use save_to_memory tool with key='plan:questions' and value='Q1: What is a monolithic kernel? Q2: What is a microkernel? Q3: What are the performance tradeoffs? Q4: Which is more secure?'\n"
            "3. Use calculator tool to compute 4 * 25 (representing 4 questions worth 25 points each)\n"
            "4. Say DONE: Plan complete. 4 research questions saved. Total points: <result>"
        ),
        priority=Priority.HIGH,
        token_budget=800,
    )
    dashboard.log_event("SPAWN", f"planner spawned", planner.pid)

    researcher = await kernel.spawn(
        "researcher",
        task=(
            "You are a researcher. Do this in order:\n"
            "1. Use read_memory tool with key='plan:questions' to get the research questions\n"
            "2. Use save_to_memory with key='research:findings' and value='Monolithic kernels run all OS services in kernel space for speed. Microkernels run services in user space for stability. Monolithic kernels (Linux) have better performance. Microkernels (QNX) have better fault isolation. Security is better in microkernels due to smaller attack surface.'\n"
            "3. Use word_count tool on 'Monolithic kernels run all OS services in kernel space for speed microkernels run services in user space'\n"
            "4. Say DONE: Research complete. Findings saved. Word count: <N>"
        ),
        priority=Priority.NORMAL,
        token_budget=800,
    )
    dashboard.log_event("SPAWN", f"researcher spawned", researcher.pid)

    writer = await kernel.spawn(
        "writer",
        task=(
            "You are a technical writer. Do this in order:\n"
            "1. Use read_memory with key='research:findings' to get the findings\n"
            "2. Use save_to_memory with key='report:draft' and value='REPORT: Microkernel vs Monolithic OS. Monolithic kernels like Linux run all services in kernel space, offering high performance but reduced fault tolerance. Microkernels like QNX isolate services in user space, improving stability and security at some performance cost. For general computing, monolithic wins on speed. For safety-critical systems, microkernels win on reliability.'\n"
            "3. Use get_time tool to timestamp the report\n"
            "4. Say DONE: Report written and saved at <timestamp>"
        ),
        priority=Priority.NORMAL,
        token_budget=800,
        parent_pid=researcher.pid,
    )
    dashboard.log_event("SPAWN", f"writer spawned", writer.pid)

    critic = await kernel.spawn(
        "critic",
        task=(
            "You are a critical reviewer. Do this in order:\n"
            "1. Use read_memory with key='report:draft' to read the report\n"
            "2. Use calculator to compute 85 + 10 (base score + bonus for covering both architectures)\n"
            "3. Use save_to_memory with key='review:score' and value='95/100 - Covers both architectures clearly'\n"
            "4. Say DONE: Review complete. Score: <score>/100"
        ),
        priority=Priority.LOW,
        token_budget=800,
        parent_pid=writer.pid,
    )
    dashboard.log_event("SPAWN", f"critic spawned", critic.pid)

    print(f"\n{'='*60}")
    print(f"  Pipeline agents spawned. Open your browser:")
    print(f"  http://localhost:8000")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------
    # Wait for all agents
    # ----------------------------------------------------------------

    results = await asyncio.gather(
        kernel.wait(planner.pid, timeout=180),
        kernel.wait(researcher.pid, timeout=180),
        kernel.wait(writer.pid, timeout=180),
        kernel.wait(critic.pid, timeout=180),
        return_exceptions=True,
    )

    # Log completions
    for proc in results:
        if not isinstance(proc, Exception):
            dashboard.log_event("DONE", f"{proc.name}: {str(proc.result or proc.error)[:50]}", proc.pid)

    # ----------------------------------------------------------------
    # Print final pipeline output
    # ----------------------------------------------------------------

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")

    # Read final memory state
    for key in ["plan:topic", "plan:questions", "research:findings", "report:draft", "review:score"]:
        entry = await kernel.memory.read(key, reader_pid="main")
        if entry:
            print(f"\n  [{key}]")
            print(f"  {str(entry.value)[:120]}")

    print(f"\n{'='*60}")
    print("  Agent Results")
    print(f"{'='*60}")
    for proc in results:
        if isinstance(proc, Exception):
            print(f"\n  ERROR: {proc}")
            continue
        status = "✓" if proc.result else "✗"
        print(f"  {status} [{proc.name}] {proc.priority.name} | {proc.budget.used} tokens | {proc.context_switches} switches")
        if proc.result:
            print(f"    → {proc.result[:80]}")
        if proc.error:
            print(f"    ✗ {proc.error}")

    stats = kernel.stats
    print(f"\n{'='*60}")
    print(f"  Kernel Stats")
    print(f"  Ticks: {stats.tick} | Switches: {stats.context_switches} | Tokens: {stats.total_tokens_used} | LLM calls: {stats.total_llm_calls} | Uptime: {stats.uptime:.1f}s")
    print(f"{'='*60}\n")

    print("  Dashboard still running at http://localhost:8000")
    print("  Press Ctrl+C to stop.\n")


async def main():
    print(f"\n{'='*60}")
    print("  Agent OS — Grand Finale (Week 5)")
    print("  Multi-agent pipeline + Live Dashboard")
    print(f"{'='*60}\n")
    print("  Starting web dashboard on http://localhost:8000 ...")
    print("  Open that URL in your browser now!\n")

    async with Kernel.boot_context(
        use_ollama=True,
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL,
        scheduling_policy=SchedulingPolicy.PRIORITY_PREEMPTIVE,
    ) as kernel:

        dashboard = DashboardServer(kernel, broadcast_interval=0.5)

        # Start uvicorn in background
        config = uvicorn.Config(
            dashboard.app,
            host="0.0.0.0",
            port=8000,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        broadcast_task = asyncio.create_task(dashboard.broadcast_loop())

        # Give server a moment to start
        await asyncio.sleep(1.0)

        try:
            await run_pipeline(kernel, dashboard)
            # Keep dashboard alive after pipeline finishes
            await asyncio.sleep(300)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            server_task.cancel()
            broadcast_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Shutting down.")
