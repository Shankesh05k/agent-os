"""
Agent OS — Memory Manager (Week 3)
Shared memory store that agents can read and write.
Two layers:
  1. KV Store   — fast key/value, backed by Redis
  2. Scratchpad — per-agent working memory (in-process dict)

Access control: owner pid, read-only pids, or public.
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger("agent_os.memory_manager")


class MemoryAccess(Enum):
    PRIVATE = "private"     # only owner can read/write
    SHARED = "shared"       # any agent can read, only owner can write
    PUBLIC = "public"       # any agent can read and write


@dataclass
class MemoryEntry:
    key: str
    value: Any
    owner_pid: str
    access: MemoryAccess = MemoryAccess.SHARED
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    ttl: Optional[int] = None   # seconds, None = forever
    tags: list[str] = field(default_factory=list)


class MemoryManager:
    """
    Shared memory for agents. Think of it as shared memory segments in an OS.

    Usage:
        # Agent writes a result
        await memory.write("research:findings", data, owner_pid=pid_a)

        # Another agent reads it
        entry = await memory.read("research:findings", reader_pid=pid_b)

        # Agent appends to a list
        await memory.append("shared:log", "step 1 done", owner_pid=pid_a)

        # Search by tag
        entries = await memory.search_by_tag("research")
    """

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client
        self._local: dict[str, MemoryEntry] = {}   # hot cache
        self._stats = {"reads": 0, "writes": 0, "denials": 0}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def write(
        self,
        key: str,
        value: Any,
        owner_pid: str,
        *,
        access: MemoryAccess = MemoryAccess.SHARED,
        ttl: int | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        existing = self._local.get(key)
        if existing and existing.access == MemoryAccess.PRIVATE:
            if existing.owner_pid != owner_pid:
                self._stats["denials"] += 1
                raise PermissionError(f"pid={owner_pid} cannot write private key '{key}' owned by {existing.owner_pid}")

        entry = MemoryEntry(
            key=key,
            value=value,
            owner_pid=owner_pid,
            access=access,
            ttl=ttl,
            tags=tags or [],
            updated_at=time.monotonic(),
        )
        self._local[key] = entry
        await self._persist(entry)
        self._stats["writes"] += 1
        logger.debug("MEMORY WRITE key=%r by pid=%s", key, owner_pid)
        return entry

    async def read(
        self,
        key: str,
        reader_pid: str,
    ) -> Optional[MemoryEntry]:
        entry = self._local.get(key)
        if not entry:
            entry = await self._load(key)
        if not entry:
            return None

        if entry.access == MemoryAccess.PRIVATE and entry.owner_pid != reader_pid:
            self._stats["denials"] += 1
            raise PermissionError(f"pid={reader_pid} cannot read private key '{key}'")

        self._stats["reads"] += 1
        return entry

    async def delete(self, key: str, requester_pid: str) -> bool:
        entry = self._local.get(key)
        if not entry:
            return False
        if entry.access == MemoryAccess.PRIVATE and entry.owner_pid != requester_pid:
            raise PermissionError(f"pid={requester_pid} cannot delete private key '{key}'")
        del self._local[key]
        await self._redis.delete(f"agent_os:mem:{key}")
        return True

    async def append(
        self,
        key: str,
        item: Any,
        owner_pid: str,
        *,
        access: MemoryAccess = MemoryAccess.SHARED,
    ):
        """Append an item to a list stored at key."""
        existing = await self.read(key, owner_pid)
        current = existing.value if existing else []
        if not isinstance(current, list):
            current = [current]
        current.append(item)
        return await self.write(key, current, owner_pid, access=access)

    # ------------------------------------------------------------------
    # Search / list
    # ------------------------------------------------------------------

    def list_keys(self, reader_pid: str) -> list[str]:
        """List all keys readable by this pid."""
        result = []
        for key, entry in self._local.items():
            if entry.access != MemoryAccess.PRIVATE or entry.owner_pid == reader_pid:
                result.append(key)
        return sorted(result)

    def search_by_tag(self, tag: str, reader_pid: str) -> list[MemoryEntry]:
        results = []
        for entry in self._local.values():
            if tag in entry.tags:
                if entry.access != MemoryAccess.PRIVATE or entry.owner_pid == reader_pid:
                    results.append(entry)
        return results

    def search_by_owner(self, owner_pid: str) -> list[MemoryEntry]:
        return [e for e in self._local.values() if e.owner_pid == owner_pid]

    # ------------------------------------------------------------------
    # Scratchpad — per-agent ephemeral working memory
    # ------------------------------------------------------------------

    def scratch_set(self, pid: str, key: str, value: Any):
        """Fast in-memory scratchpad for an agent's working notes."""
        scratch_key = f"__scratch__{pid}__{key}"
        self._local[scratch_key] = MemoryEntry(
            key=scratch_key, value=value, owner_pid=pid,
            access=MemoryAccess.PRIVATE,
        )

    def scratch_get(self, pid: str, key: str) -> Any:
        scratch_key = f"__scratch__{pid}__{key}"
        entry = self._local.get(scratch_key)
        return entry.value if entry else None

    def scratch_clear(self, pid: str):
        keys_to_delete = [k for k in self._local if k.startswith(f"__scratch__{pid}__")]
        for k in keys_to_delete:
            del self._local[k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "total_keys": len([k for k in self._local if not k.startswith("__scratch__")]),
            "total_scratch_keys": len([k for k in self._local if k.startswith("__scratch__")]),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(self, entry: MemoryEntry):
        try:
            data = json.dumps({
                "key": entry.key,
                "value": entry.value if isinstance(entry.value, (str, int, float, bool, list, dict)) else str(entry.value),
                "owner_pid": entry.owner_pid,
                "access": entry.access.value,
                "tags": entry.tags,
                "updated_at": entry.updated_at,
            })
            redis_key = f"agent_os:mem:{entry.key}"
            if entry.ttl:
                await self._redis.set(redis_key, data, ex=entry.ttl)
            else:
                await self._redis.set(redis_key, data)
        except Exception as e:
            logger.debug("Memory persist failed: %s", e)

    async def _load(self, key: str) -> Optional[MemoryEntry]:
        try:
            data = await self._redis.get(f"agent_os:mem:{key}")
            if not data:
                return None
            d = json.loads(data)
            entry = MemoryEntry(
                key=d["key"],
                value=d["value"],
                owner_pid=d["owner_pid"],
                access=MemoryAccess(d["access"]),
                tags=d.get("tags", []),
            )
            self._local[key] = entry
            return entry
        except Exception:
            return None
