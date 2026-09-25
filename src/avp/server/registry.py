"""Thread-safe VRAM registry for latent AVP contexts.

A single 7B KV-cache is roughly 390 MB, so contexts cannot simply accumulate.
This registry adds, on top of :class:`avp.ContextStore`:

* source-model metadata (needed to recover the source connector for cross-model
  projection),
* an entry-count cap with least-recently-used eviction,
* lazy TTL expiry with explicit accounting.

The registry only stores references to in-memory tensors.  It never serializes
them; cross-process transfer is out of scope for the latent server.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StoredContext:
    """A latent context plus the metadata required to route it.

    Attributes:
        context_id: Short opaque handle returned to the agent.
        context: The heavy :class:`avp.context.AVPContext` (tensor-backed).
        source_model_id: Model identifier that produced the context.
        source_model_hash: Model config hash; used to find the source connector.
        payload_type: ``"KV_CACHE"`` or ``"HIDDEN_STATE"`` as a string.
        num_steps: Latent thinking steps that produced the context.
        seq_len: KV-cache sequence length.
        created_at: ``time.time()`` when stored.
        last_access: ``time.time()`` of the most recent :meth:`get`.
        ttl: Time-to-live in seconds from ``created_at``.
    """

    context_id: str
    context: Any
    source_model_id: str
    source_model_hash: str
    payload_type: str
    num_steps: int
    seq_len: int
    created_at: float
    last_access: float
    ttl: float

    @property
    def is_expired(self) -> bool:
        """Whether the entry has outlived its TTL."""
        return time.time() > self.created_at + self.ttl

    @property
    def size_bytes(self) -> int:
        """Best-effort in-memory size of the stored KV-cache, in bytes.

        Returns 0 when the payload is not a torch tensor (e.g. hidden-state
        only, or a mock in tests).
        """
        pkv = getattr(self.context, "past_key_values", None)
        if pkv is None:
            return 0
        total = 0
        try:
            for layer in pkv:
                for tensor in layer:
                    if tensor is None:
                        continue
                    total += int(tensor.numel()) * int(tensor.element_size())
        except TypeError:
            return 0
        return total

    def to_public(self) -> dict[str, Any]:
        """Return a JSON-serializable summary (no tensors)."""
        return {
            "context_id": self.context_id,
            "source_model_id": self.source_model_id,
            "source_model_hash": self.source_model_hash,
            "payload_type": self.payload_type,
            "num_steps": self.num_steps,
            "seq_len": self.seq_len,
            "ttl": self.ttl,
            "age_s": round(time.time() - self.created_at, 3),
            "idle_s": round(time.time() - self.last_access, 3),
            "size_bytes": self.size_bytes,
        }


class ContextRegistry:
    """Thread-safe store of latent contexts with TTL and LRU eviction.

    Args:
        default_ttl: Default per-entry TTL in seconds.  A context being reused
            still expires ``default_ttl`` after it was created; callers that
            need a longer-lived context should re-think or raise the TTL.
        max_entries: Maximum number of live contexts.  When full, expired
            entries are removed first; if none are expired the least-recently
            used entry is evicted.

    All public methods are safe to call from multiple threads.
    """

    def __init__(self, default_ttl: float = 300.0, max_entries: int = 8) -> None:
        if default_ttl <= 0:
            raise ValueError(f"default_ttl must be > 0, got {default_ttl}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._entries: dict[str, StoredContext] = {}
        self._lock = threading.Lock()
        self._default_ttl = float(default_ttl)
        self._max_entries = int(max_entries)

    def put(
        self,
        context: Any,
        source_model_id: str = "",
        source_model_hash: str = "",
        ttl: float | None = None,
        context_id: str | None = None,
    ) -> StoredContext:
        """Store a context and return its :class:`StoredContext` handle.

        Args:
            context: The :class:`avp.context.AVPContext` to store.
            source_model_id: Model identifier that produced the context.
            source_model_hash: Model config hash for the source connector.
            ttl: Per-entry TTL.  ``None`` uses the registry default.
            context_id: Explicit handle.  Mostly useful for tests; a random
                UUID4 hex is generated when omitted.

        Returns:
            The stored entry (its ``context_id`` is the agent-facing handle).
        """
        now = time.time()
        entry = StoredContext(
            context_id=context_id or uuid.uuid4().hex,
            context=context,
            source_model_id=source_model_id,
            source_model_hash=source_model_hash or getattr(context, "model_hash", ""),
            payload_type=getattr(getattr(context, "payload_type", None), "name", "UNKNOWN"),
            num_steps=int(getattr(context, "num_steps", 0) or 0),
            seq_len=int(getattr(context, "seq_len", 0) or 0),
            created_at=now,
            last_access=now,
            ttl=float(ttl) if ttl is not None else self._default_ttl,
        )
        with self._lock:
            self._evict_locked()
            self._entries[entry.context_id] = entry
        logger.info(
            "registry: stored context %s (model=%s, payload=%s, ttl=%.0fs)",
            entry.context_id, source_model_id or "?", entry.payload_type, entry.ttl,
        )
        return entry

    def get(self, context_id: str, touch: bool = True) -> StoredContext | None:
        """Look up a context, removing it lazily if expired.

        Args:
            context_id: Handle returned by :meth:`put`.
            touch: Update ``last_access`` for LRU accounting (default True).

        Returns:
            The :class:`StoredContext`, or ``None`` if missing/expired.
        """
        with self._lock:
            entry = self._entries.get(context_id)
            if entry is None:
                return None
            if entry.is_expired:
                age = time.time() - entry.created_at
                del self._entries[context_id]
                logger.warning(
                    "registry: context %s expired (age=%.1fs, ttl=%.1fs)",
                    context_id, age, entry.ttl,
                )
                return None
            if touch:
                entry.last_access = time.time()
            return entry

    def peek(self, context_id: str) -> StoredContext | None:
        """Like :meth:`get` but does not update LRU recency."""
        return self.get(context_id, touch=False)

    def remove(self, context_id: str) -> bool:
        """Remove an entry, freeing its KV-cache.  Returns True if it existed."""
        with self._lock:
            return self._entries.pop(context_id, None) is not None

    # Alias used by the tool surface.
    release = remove

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries.  Returns the number removed."""
        with self._lock:
            expired = [k for k, e in self._entries.items() if e.is_expired]
            for key in expired:
                del self._entries[key]
            return len(expired)

    def keys(self) -> list[str]:
        """Return handles of all live (non-expired) entries."""
        self.cleanup_expired()
        with self._lock:
            return list(self._entries.keys())

    def __contains__(self, context_id: object) -> bool:
        """Whether a live context exists for ``context_id``."""
        if not isinstance(context_id, str):
            return False
        return self.get(context_id, touch=False) is not None

    def __iter__(self) -> Iterator[str]:
        """Iterate over live context handles."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Number of live (non-expired) contexts."""
        return self.active_count

    @property
    def active_count(self) -> int:
        """Number of live (non-expired) entries."""
        self.cleanup_expired()
        with self._lock:
            return len(self._entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def default_ttl(self) -> float:
        return self._default_ttl

    def stats(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the registry."""
        self.cleanup_expired()
        with self._lock:
            entries = list(self._entries.values())
        total_bytes = sum(e.size_bytes for e in entries)
        return {
            "active_count": len(entries),
            "max_entries": self._max_entries,
            "default_ttl": self._default_ttl,
            "total_size_bytes": total_bytes,
            "contexts": [e.to_public() for e in entries],
        }

    def _evict_locked(self) -> None:
        """Drop expired entries, then LRU entries, until below capacity.

        Caller must hold ``self._lock``.
        """
        if len(self._entries) < self._max_entries:
            return
        expired = [k for k, e in self._entries.items() if e.is_expired]
        for key in expired:
            del self._entries[key]
            logger.info("registry: evicted expired context %s", key)
        while len(self._entries) >= self._max_entries:
            victim = min(self._entries.values(), key=lambda e: e.last_access)
            del self._entries[victim.context_id]
            logger.warning(
                "registry: capacity reached (%d); LRU-evicted context %s "
                "(model=%s)",
                self._max_entries, victim.context_id, victim.source_model_id or "?",
            )
