"""
Agent OS — Core Data Models
Agents are processes. Treat them like an OS treats processes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import time
import uuid


class AgentState(Enum):
    READY = "ready"          # Queued, waiting for CPU slot
    RUNNING = "running"      # Actively consuming tokens / executing
    BLOCKED = "blocked"      # Waiting on IPC message or tool result
    SLEEPING = "sleeping"    # Voluntarily paused, waiting on timer
    ZOMBIE = "zombie"        # Finished but result not yet collected
    DEAD = "dead"            # Collected / terminated


class Priority(Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2
    BACKGROUND = 3


@dataclass
class TokenBudget:
    """Token accounting for a single agent process."""
    total_allocated: int
    used: int = 0
    reserved: int = 0      # Reserved for current in-flight call

    @property
    def available(self) -> int:
        return self.total_allocated - self.used - self.reserved

    @property
    def utilization(self) -> float:
        return self.used / self.total_allocated if self.total_allocated else 0.0

    def reserve(self, amount: int) -> bool:
        if amount > self.available:
            return False
        self.reserved += amount
        return True

    def commit(self, actual_used: int):
        """Call after an LLM response arrives with the real token count."""
        self.used += actual_used
        self.reserved = 0

    def release_reservation(self):
        self.reserved = 0


@dataclass
class AgentProcess:
    """
    The process descriptor — the kernel's view of a running agent.
    Analogous to a PCB (Process Control Block) in a real OS.
    """
    pid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "unnamed"
    state: AgentState = AgentState.READY
    priority: Priority = Priority.NORMAL

    # Token management
    budget: TokenBudget = field(default_factory=lambda: TokenBudget(total_allocated=4096))

    # Scheduling metadata
    created_at: float = field(default_factory=time.monotonic)
    last_scheduled_at: Optional[float] = None
    cpu_time: float = 0.0           # Wall-clock seconds spent "running"
    context_switches: int = 0

    # Agent config — passed to the LLM on each scheduling tick
    system_prompt: str = ""
    task_description: str = ""
    tools: list[str] = field(default_factory=list)

    # Conversation memory (short-term, kept in process)
    message_history: list[dict] = field(default_factory=list)

    # IPC inbox — messages from other agents
    inbox: list[IPCMessage] = field(default_factory=list)

    # Output / result when ZOMBIE
    result: Optional[Any] = None
    error: Optional[str] = None

    # Parent/child relationships
    parent_pid: Optional[str] = None
    children: list[str] = field(default_factory=list)

    # Metadata bag for user-defined tags
    tags: dict[str, str] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"AgentProcess(pid={self.pid}, name={self.name!r}, "
            f"state={self.state.value}, priority={self.priority.name}, "
            f"tokens={self.budget.used}/{self.budget.total_allocated})"
        )


@dataclass
class IPCMessage:
    """
    A message on the inter-agent communication bus.
    Loosely modelled on POSIX signals + message queues.
    """
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_pid: str = ""
    recipient_pid: str = ""        # "" means broadcast
    channel: str = "default"
    payload: Any = None
    timestamp: float = field(default_factory=time.monotonic)
    reply_to: Optional[str] = None  # msg_id to correlate replies

    def __repr__(self):
        return (
            f"IPCMessage(from={self.sender_pid}→{self.recipient_pid}, "
            f"channel={self.channel!r}, id={self.msg_id})"
        )


@dataclass
class SchedulerStats:
    """Live telemetry from the scheduler."""
    tick: int = 0
    total_agents_spawned: int = 0
    total_agents_dead: int = 0
    total_tokens_used: int = 0
    total_llm_calls: int = 0
    total_ipc_messages: int = 0
    context_switches: int = 0
    uptime_start: float = field(default_factory=time.monotonic)

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.uptime_start
