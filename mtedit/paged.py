"""Paged prefix K/V store built on SGLang's LLM prefix-cache machinery.

* index: ``sglang.srt.mem_cache.radix_cache.RadixCache`` (token-granular radix
  tree with node splitting, lock refs and leaf-first LRU eviction);
* slots: ``sglang.srt.mem_cache.allocator.token.TokenToKVPoolAllocator`` over
  one preallocated pool, like the LLM ``TokenToKVPool``: no per-request
  allocations, no fragmentation, eviction just returns slot indices.

Keys are *virtual token ids* (``keys.virtual_ids``): one 63-bit id per prefix
position, chained so equal ids imply equal K/V. Text positions chain per token,
so a partially shared prompt reuses its shared head; every position of an image
block includes the image's content key, so a block matches whole or not at all
(image tokens attend bidirectionally within their block).

A pool holds ``[layers, 2, slots, heads, head_dim]`` K/V plus optional
per-slot extra rows (the text encoder stores its last hidden state there).
"""

from __future__ import annotations

import logging
import threading
from array import array
from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

logger = logging.getLogger(__name__)


@dataclass
class PoolStats:
    lookups: int = 0
    hit_tokens: int = 0
    miss_tokens: int = 0
    inserted_tokens: int = 0
    evicted_tokens: int = 0
    contiguous_views: int = 0
    gathers: int = 0


class Match:
    """A locked prefix match; release it (or insert through it) when done."""

    def __init__(self, pool: "PagedKVPool", length: int, indices: torch.Tensor, node):
        self.pool, self.length, self.indices, self.node = pool, length, indices, node

    def release(self) -> None:
        if self.node is not None:
            self.pool.tree.dec_lock_ref(self.node)
            self.node = None


class PagedKVPool:
    def __init__(self, name, layers, heads, head_dim, budget_bytes, dtype, device, extra_dim=0):
        per_token = layers * 2 * heads * head_dim * torch.empty((), dtype=dtype).element_size()
        per_token += extra_dim * torch.empty((), dtype=dtype).element_size()
        self.name = name
        self.capacity = max(int(budget_bytes // per_token), 1)
        self.layers, self.heads, self.head_dim = layers, heads, head_dim
        # slot 0 is the allocator's reserved padding slot
        self.kv = torch.empty(layers, 2, self.capacity + 1, heads, head_dim, dtype=dtype, device=device)
        self.extra = (
            torch.empty(self.capacity + 1, extra_dim, dtype=dtype, device=device) if extra_dim else None
        )
        self.allocator = TokenToKVPoolAllocator(
            self.capacity, dtype=dtype, device=str(device), kvcache=None, need_sort=True
        )
        self.tree = RadixCache.create_simulated(mock_allocator=self.allocator, page_size=1)
        self.stats = PoolStats()
        self.lock = threading.RLock()
        logger.info(
            "mtedit: %s pool: %d tokens, %.1f GiB (%d B/token)",
            name, self.capacity, self.capacity * per_token / (1 << 30), per_token,
        )

    # lookup / insert -----------------------------------------------------------

    @staticmethod
    def _key(vids) -> RadixKey:
        if isinstance(vids, array):
            return RadixKey(vids)
        ids = array("q")
        ids.frombytes(np.ascontiguousarray(vids, dtype=np.int64).tobytes())
        return RadixKey(ids)

    def match(self, vids) -> Match:
        with self.lock:
            self.stats.lookups += 1
            result = self.tree.match_prefix(MatchPrefixParams(key=self._key(vids)))
            length = result.device_prefix_len
            node = result.last_device_node
            if length:
                self.tree.inc_lock_ref(node)
                indices = self.tree.path_device_indices(node)[:length]
            else:
                node, indices = None, None
            self.stats.hit_tokens += length
            self.stats.miss_tokens += len(vids) - length
            return Match(self, length, indices, node)

    def alloc(self, n: int) -> torch.Tensor | None:
        with self.lock:
            if n > self.capacity:
                return None
            if self.allocator.available_size() < n:
                need = n - self.allocator.available_size()
                evicted = self.tree.evict(EvictParams(num_tokens=need)).num_tokens_evicted
                self.stats.evicted_tokens += evicted
            return self.allocator.alloc(n)

    def insert(self, vids, indices: torch.Tensor, owned_from: int) -> None:
        """Insert the full key. ``indices[owned_from:]`` were allocated by the caller;
        any of those whose position turned out to be present already are freed."""
        with self.lock:
            result = self.tree.insert(InsertParams(key=self._key(vids), value=indices.to(torch.int64)))
            present = result.prefix_len
            if present > owned_from:
                self.allocator.free(indices[owned_from:present])
            self.stats.inserted_tokens += len(vids) - max(present, owned_from)

    # tensor views ----------------------------------------------------------------

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """``[layers, 2, n, heads, head_dim]`` for ``indices``: a view when they are one ascending run."""
        n = indices.numel()
        first = int(indices[0]) if n else 0
        # one host sync per request; ascending runs come straight from a fresh pool
        if n and int(indices[-1]) - first == n - 1 and (n == 1 or bool((indices[1:] - indices[:-1] == 1).all())):
            self.stats.contiguous_views += 1
            return self.kv[:, :, first : first + n]
        self.stats.gathers += 1
        return self.kv.index_select(2, indices)

    def write(self, indices: torch.Tensor, kv: torch.Tensor) -> None:
        """Store ``kv`` (``[layers, 2, n, heads, head_dim]``) at ``indices``."""
        self.kv.index_copy_(2, indices, kv.to(self.kv.dtype))

    def usage(self) -> dict:
        with self.lock:
            return dict(
                capacity=self.capacity,
                free=self.allocator.available_size(),
                evictable=self.tree.evictable_size(),
                protected=self.tree.protected_size(),
                **self.stats.__dict__,
            )

    def clear(self) -> None:
        with self.lock:
            self.tree.reset()
            self.allocator.clear()
