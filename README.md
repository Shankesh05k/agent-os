# Agent OS

> An operating system for LLM agents. Not one agent — the infrastructure layer that runs them.

---

## The idea

Most agent frameworks ask: *what should this agent do?*  
Agent OS asks: *how do multiple agents share compute, memory, and communicate — just like an OS manages processes?*

Every agent is a **process**. The kernel schedules it, enforces its token budget, routes messages to its inbox, and collects its result when it's done.

```
┌───────────────────────────────────────────────────┐
│                   Agent OS Kernel                 │
│                                                   │
│   ┌───────────┐  ┌────────────┐  ┌─────────────┐ │
│   │ Scheduler │  │  Process   │  │   IPC Bus   │ │
│   │ (preempt  │  │  Manager   │  │  (msg queue)│ │
│   │  / r-r)   │  │            │  │             │ │
│   └───────────┘  └────────────┘  └─────────────┘ │
│   ┌───────────┐  ┌────────────┐                   │
│   │   Token   │  │    LLM     │                   │
│   │  Budget   │  │  Executor  │                   │
│   └───────────┘  └────────────┘                   │
└───────────────────────────────────────────────────┘
        │                │               │
    Agent A          Agent B          Agent C
  (HIGH prio)     (NORMAL prio)     (LOW prio)
```

---

## Subsystems

| Subsystem | File | OS analogue |
|---|---|---|
| Process Manager | `kernel/process_manager.py` | PCB table, fork/wait/kill |
| Scheduler | `kernel/scheduler.py` | CPU scheduler, context switch |
| IPC Bus | `kernel/ipc.py` | Message queues, signals |
| Token Budget | `kernel/models.py` | Memory quota / cgroups |
| LLM Executor | `kernel/executor.py` | CPU execution unit |
| Kernel facade | `kernel/kernel.py` | syscall interface |

---

## Quickstart

```bash
pip install -r requirements.txt

# Start Redis (required for process table persistence)
docker run -d -p 6379:6379 redis:alpine

# Set your API key
export ANTHROPIC_API_KEY=sk-...

# Run the multi-agent demo
python -m examples.multi_agent_demo
```

---

## Usage

```python
from kernel import Kernel, Priority, SchedulingPolicy

async with Kernel.boot_context(scheduling_policy=SchedulingPolicy.PRIORITY_PREEMPTIVE) as kernel:

    # Spawn agents like processes
    analyst = await kernel.spawn(
        "analyst",
        task="Analyse the impact of DRAM supply constraints on AI chip prices. Say DONE: when finished.",
        priority=Priority.HIGH,
        token_budget=2048,
    )

    writer = await kernel.spawn(
        "writer",
        task="Write a concise report intro paragraph about semiconductor trends. Say DONE: when finished.",
        priority=Priority.NORMAL,
        token_budget=1024,
    )

    # Send a message from one agent to another
    await kernel.send_message(analyst.pid, writer.pid, "Focus on HBM3 specifically", channel="directive")

    # Wait for results (blocks until done)
    result = await kernel.wait(analyst.pid, timeout=60)
    print(result.result)

    # Inspect the process table
    for proc in kernel.ps():
        print(proc)
```

---

## Agent state machine

```
READY → RUNNING → ZOMBIE → DEAD
          ↓  ↑
        BLOCKED
          ↓  ↑
        SLEEPING
```

| State | Meaning |
|---|---|
| `READY` | Queued, waiting for a scheduler slot |
| `RUNNING` | Actively executing (LLM call in-flight) |
| `BLOCKED` | Waiting on an IPC message or tool result |
| `SLEEPING` | Voluntarily paused with a wake timer |
| `ZOMBIE` | Finished; result available, not yet collected |
| `DEAD` | Collected or killed |

---

## Scheduling policies

**Priority preemptive** (default): Strict priority ordering — `HIGH > NORMAL > LOW > BACKGROUND`. Within the same priority, oldest-first to prevent starvation.

**Round-robin**: Each READY agent gets an equal time slice in creation order. Fairer for equal-priority workloads.

Switch at boot:
```python
Kernel(scheduling_policy=SchedulingPolicy.ROUND_ROBIN)
```

---

## Token budgeting

Each process has a `TokenBudget`:
- **allocated** — total tokens the process may use across its lifetime
- **reserved** — held for the current in-flight LLM call
- **used** — actual tokens consumed (committed after each response)

The scheduler kills any process whose budget hits zero before dispatching it.

---

## IPC

Agents communicate through the IPC Bus:

```python
# Direct message
await kernel.send_message(pid_a, pid_b, payload="here are my results", channel="result")

# Broadcast
await kernel.broadcast(pid_a, payload="shutdown signal", channel="alert")
```

Inbox messages are automatically injected as context on the agent's next scheduling tick.

---

## Week-by-week roadmap

| Week | Goal |
|---|---|
| **1** ✅ | Core kernel: Process Manager, Scheduler, IPC Bus, Token Budget |
| **2** | Tool registry — agents can call Python functions; sandboxed exec |
| **3** | Memory Manager — shared vector store agents can read/write |
| **4** | Dashboard — live process table, token burn chart, IPC message trace |
| **5** | Agent templates + complex demo (multi-agent research pipeline) |

---

## Tests

```bash
python -m pytest tests/ -v
# 20 tests, no Redis or API key needed
```
