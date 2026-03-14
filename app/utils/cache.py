"""
Unified generic LRU cache for OptikR.

Provides a single ``LRUCache[K, V]`` with optional TTL, thread safety,
optional memory tracking, and hit/miss/eviction statistics.  Replaces the
scattered cache implementations (OCRCache, TranslationCache, SmartCache,
DictionaryLookupCache, pipeline_cache_manager.LRUCache).

Uses ``OrderedDict`` for O(1) LRU operations.

Requirements: 5.1, 5.2, 5.3
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)

K = TypeVar("K")
V = TypeVar("V")

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry(Generic[V]):
    """A single cache entry with metadata."""

    value: V
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0


class LRUCache(Generic[K, V]):
    """Generic LRU cache with optional TTL and memory tracking.

    Parameters
    ----------
    max_size:
        Maximum number of entries.  When exceeded the least-recently-used
        entry is evicted.
    ttl_seconds:
        Optional time-to-live in seconds.  Entries older than this are
        treated as expired and removed on access.
    max_memory_mb:
        Optional memory cap in megabytes.  When set, entries are evicted
        if total estimated memory exceeds this limit.  Pass ``None`` to
        disable memory tracking (default).
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float | None = None,
        max_memory_mb: float | None = None,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._max_memory_bytes: int | None = (
            int(max_memory_mb * 1024 * 1024) if max_memory_mb is not None else None
        )
        self._store: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_memory_bytes = 0

    # --- public properties ---

    @property
    def max_size(self) -> int:
        """Return the maximum capacity of the cache."""
        return self._max_size

    @property
    def ttl_seconds(self) -> float | None:
        """Return the TTL setting (``None`` means no expiration)."""
        return self._ttl

    # --- core API ---

    def get(self, key: K) -> V | None:
        """Return the value for *key*, or ``None`` if missing / expired.

        Accessing an entry marks it as most-recently-used.
        """
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None

            entry = self._store[key]

            # Check TTL expiration
            if self._ttl is not None and (time.time() - entry.created_at) > self._ttl:
                self._current_memory_bytes -= entry.size_bytes
                del self._store[key]
                self._misses += 1
                return None

            # Mark as most-recently-used
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: K, value: V, size_bytes: int = 0) -> None:
        """Insert or update *key* with *value*.

        If the cache is at capacity the least-recently-used entry is evicted.

        Parameters
        ----------
        size_bytes:
            Optional pre-computed size in bytes.  When ``0`` and memory
            tracking is enabled, the size is estimated automatically.
        """
        with self._lock:
            if size_bytes == 0 and self._max_memory_bytes is not None:
                size_bytes = self._estimate_size(value)

            # Update existing — move to end (most recent)
            if key in self._store:
                old = self._store[key]
                self._current_memory_bytes -= old.size_bytes
                self._store.move_to_end(key)
                old.value = value
                old.created_at = time.time()
                old.size_bytes = size_bytes
                self._current_memory_bytes += size_bytes
                return

            # Evict until we have room (count + memory)
            while self._should_evict(size_bytes):
                if not self._store:
                    break
                self._evict_lru()

            self._store[key] = CacheEntry(value=value, size_bytes=size_bytes)
            self._current_memory_bytes += size_bytes

    def remove(self, key: K) -> bool:
        """Remove *key* from the cache.  Return ``True`` if it existed."""
        with self._lock:
            if key in self._store:
                self._current_memory_bytes -= self._store[key].size_bytes
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def size(self) -> int:
        """Return the current number of entries."""
        with self._lock:
            return len(self._store)

    # --- statistics ---

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            stats: dict[str, Any] = {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }
            if self._ttl is not None:
                stats["ttl_seconds"] = self._ttl
            if self._max_memory_bytes is not None:
                stats["memory_mb"] = self._current_memory_bytes / (1024 * 1024)
                stats["max_memory_mb"] = self._max_memory_bytes / (1024 * 1024)
            return stats

    # --- serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cache to a plain dict."""
        with self._lock:
            entries: list[dict[str, Any]] = []
            for k, entry in self._store.items():
                entries.append(
                    {
                        "key": k,
                        "value": entry.value,
                        "created_at": entry.created_at,
                        "size_bytes": entry.size_bytes,
                    }
                )
            return {
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "max_memory_mb": (
                    self._max_memory_bytes / (1024 * 1024)
                    if self._max_memory_bytes is not None
                    else None
                ),
                "entries": entries,
            }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        value_deserializer: Callable[[Any], V] | None = None,
    ) -> LRUCache[K, V]:
        """Reconstruct a cache from the dict produced by ``to_dict()``.

        Entries are sorted by ``created_at`` to approximate original LRU
        order, and excess entries beyond ``max_size`` are truncated.
        """
        max_size = data["max_size"]
        cache: LRUCache[K, V] = cls(
            max_size=max_size,
            ttl_seconds=data.get("ttl_seconds"),
            max_memory_mb=data.get("max_memory_mb"),
        )
        # Sort by created_at so oldest entries are inserted first (LRU order)
        raw_entries = sorted(
            data.get("entries", []), key=lambda e: e.get("created_at", 0)
        )
        # Truncate to max_size — keep the most recent entries
        if len(raw_entries) > max_size:
            raw_entries = raw_entries[-max_size:]

        for item in raw_entries:
            value = item["value"]
            if value_deserializer is not None:
                value = value_deserializer(value)
            entry = CacheEntry(
                value=value,
                created_at=item["created_at"],
                size_bytes=item.get("size_bytes", 0),
            )
            cache._store[item["key"]] = entry
            cache._current_memory_bytes += entry.size_bytes
        return cache

    # --- internals ---

    def _should_evict(self, incoming_bytes: int) -> bool:
        """Return ``True`` if an eviction is needed before inserting."""
        if len(self._store) >= self._max_size:
            return True
        if (
            self._max_memory_bytes is not None
            and self._current_memory_bytes + incoming_bytes > self._max_memory_bytes
        ):
            return True
        return False

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry."""
        key, entry = self._store.popitem(last=False)
        self._current_memory_bytes -= entry.size_bytes
        self._evictions += 1
        logger.debug("Evicted cache entry: %s", key)

    @staticmethod
    def _estimate_size(obj: Any) -> int:
        """Estimate the memory footprint of *obj* in bytes."""
        if isinstance(obj, (str, bytes)):
            return len(obj)
        if isinstance(obj, (list, tuple)):
            return sum(LRUCache._estimate_size(item) for item in obj)
        if isinstance(obj, dict):
            return sum(
                LRUCache._estimate_size(k) + LRUCache._estimate_size(v)
                for k, v in obj.items()
            )
        # numpy arrays
        nbytes = getattr(obj, "nbytes", None)
        if nbytes is not None:
            return nbytes
        return sys.getsizeof(obj)
