"""
Agent OS — IPC Bus
Inter-Process Communication for agents. Supports:
  - Direct messages (pid → pid)
  - Broadcast (pid → all)
  - Named channels / topics
  - Request-reply pattern

Backed by Redis pub/sub for cross-process delivery + in-memory queues
for agents within the same process space.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Optional

import redis.asyncio as aioredis

from .models import IPCMessage, AgentProcess, AgentState

logger = logging.getLogger("agent_os.ipc")


class IPCBus:
    """
    Message bus for agent-to-agent communication.

    Usage:
        await bus.send(sender_pid, recipient_pid, payload, channel="result")
        msg = await bus.receive(my_pid, timeout=5.0)
        await bus.broadcast(sender_pid, payload, channel="alert")
    """

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client
        self._local_queues: dict[str, asyncio.Queue] = {}
        self._stats_sent = 0
        self._stats_dropped = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, pid: str):
        """Register a process so it can receive messages."""
        if pid not in self._local_queues:
            self._local_queues[pid] = asyncio.Queue(maxsize=256)

    def unregister(self, pid: str):
        self._local_queues.pop(pid, None)

    async def send(
        self,
        sender_pid: str,
        recipient_pid: str,
        payload,
        *,
        channel: str = "default",
        reply_to: str | None = None,
    ) -> IPCMessage:
        msg = IPCMessage(
            sender_pid=sender_pid,
            recipient_pid=recipient_pid,
            channel=channel,
            payload=payload,
            reply_to=reply_to,
        )
        await self._deliver(msg)
        self._stats_sent += 1
        logger.debug("IPC SEND %s", msg)
        return msg

    async def broadcast(
        self,
        sender_pid: str,
        payload,
        *,
        channel: str = "broadcast",
        exclude: list[str] | None = None,
    ) -> int:
        """Send to all registered processes. Returns delivery count."""
        exclude_set = set(exclude or []) | {sender_pid}
        targets = [pid for pid in self._local_queues if pid not in exclude_set]
        for pid in targets:
            await self.send(sender_pid, pid, payload, channel=channel)
        return len(targets)

    async def receive(
        self,
        pid: str,
        *,
        channel: str | None = None,
        timeout: float = 0.0,
    ) -> Optional[IPCMessage]:
        """
        Pop the next message from the pid's inbox.
        If channel is specified, drain the queue looking for a match.
        timeout=0 → non-blocking peek.
        """
        queue = self._local_queues.get(pid)
        if queue is None:
            raise KeyError(f"PID {pid} not registered on IPC bus")

        if timeout > 0:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
        else:
            try:
                msg = queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

        if channel and msg.channel != channel:
            # Put it back and return nothing — simple channel filter
            await queue.put(msg)
            return None

        return msg

    async def receive_all(self, pid: str) -> list[IPCMessage]:
        """Drain all pending messages (non-blocking)."""
        queue = self._local_queues.get(pid)
        if not queue:
            return []
        msgs = []
        while True:
            try:
                msgs.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    async def reply(self, original: IPCMessage, sender_pid: str, payload) -> IPCMessage:
        """Send a reply correlated to an original message."""
        return await self.send(
            sender_pid,
            original.sender_pid,
            payload,
            channel=f"{original.channel}.reply",
            reply_to=original.msg_id,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "registered_pids": len(self._local_queues),
            "messages_sent": self._stats_sent,
            "messages_dropped": self._stats_dropped,
            "queue_depths": {pid: q.qsize() for pid, q in self._local_queues.items()},
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _deliver(self, msg: IPCMessage):
        queue = self._local_queues.get(msg.recipient_pid)
        if queue is None:
            logger.warning("IPC DROP: recipient %s not registered", msg.recipient_pid)
            self._stats_dropped += 1
            return

        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            logger.warning("IPC DROP: queue full for pid=%s", msg.recipient_pid)
            self._stats_dropped += 1

        # Mirror to Redis for persistence / cross-host delivery
        await self._redis_publish(msg)

    async def _redis_publish(self, msg: IPCMessage):
        try:
            data = json.dumps({
                "msg_id": msg.msg_id,
                "sender": msg.sender_pid,
                "recipient": msg.recipient_pid,
                "channel": msg.channel,
                "timestamp": msg.timestamp,
                "reply_to": msg.reply_to,
                # payload not serialised here (may not be JSON-safe)
            })
            channel_key = f"agent_os:ipc:{msg.recipient_pid}"
            await self._redis.lpush(channel_key, data)
            await self._redis.expire(channel_key, 300)
        except Exception as e:
            logger.debug("IPC Redis publish failed: %s", e)
