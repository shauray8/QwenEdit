"""Cross-request prefix K/V store: a radix tree over segment chain keys.

Each node holds one segment's K/V for every layer as a single tensor
``[layers, 2, tokens, heads, head_dim]``. A request's prefix K/V is the
concatenation of the nodes along its chain, so a session's anchor image is
stored once and shared by every later turn, re-roll and CFG branch.

Two tiers: GPU nodes over ``gpu_budget`` move to pinned host memory (least
recently used first) and come back on the next hit; host nodes over
``cpu_budget`` are dropped together with their descendants, which can no longer
be reached.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

GIB = 1 << 30


@dataclass(eq=False)
class Node:
    key: bytes
    parent: "Node | None"
    kv: torch.Tensor
    last_used: float = field(default_factory=time.monotonic)
    children: dict[bytes, "Node"] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return self.kv.numel() * self.kv.element_size()

    @property
    def tokens(self) -> int:
        return self.kv.shape[2]


@dataclass
class StoreStats:
    lookups: int = 0
    hit_segments: int = 0
    miss_segments: int = 0
    hit_tokens: int = 0
    miss_tokens: int = 0
    inserted_tokens: int = 0
    offloads: int = 0
    reloads: int = 0
    drops: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PrefixKVStore:
    def __init__(self, gpu_budget: int, cpu_budget: int):
        self.gpu_budget = gpu_budget
        self.cpu_budget = cpu_budget
        self.roots: dict[bytes, Node] = {}
        self.nodes: dict[bytes, Node] = {}
        self.stats = StoreStats()
        self._lock = threading.RLock()

    # accounting -------------------------------------------------------------

    def _bytes_on(self, device_type: str) -> int:
        return sum(n.nbytes for n in self.nodes.values() if n.kv.device.type == device_type)

    def usage(self) -> dict:
        with self._lock:
            return dict(
                nodes=len(self.nodes),
                gpu_gib=self._bytes_on("cuda") / GIB,
                cpu_gib=self._bytes_on("cpu") / GIB,
                **self.stats.as_dict(),
            )

    # lookup / insert ---------------------------------------------------------

    def match(self, chain: list[bytes], device: torch.device) -> list[Node]:
        """Longest stored prefix of ``chain``; matched nodes are moved to ``device``."""
        with self._lock:
            self.stats.lookups += 1
            matched: list[Node] = []
            children = self.roots
            for key in chain:
                node = children.get(key)
                if node is None:
                    break
                matched.append(node)
                children = node.children
            now = time.monotonic()
            for node in matched:
                node.last_used = now
                if node.kv.device != device:
                    node.kv = node.kv.to(device, non_blocking=True)
                    self.stats.reloads += 1
            self.stats.hit_segments += len(matched)
            self.stats.miss_segments += len(chain) - len(matched)
            return matched

    def insert(self, parent: Node | None, key: bytes, kv: torch.Tensor) -> Node:
        """Add one segment under ``parent`` (None: a root segment); ``kv`` is owned by the store."""
        with self._lock:
            siblings = parent.children if parent is not None else self.roots
            node = siblings.get(key)
            if node is None:
                node = Node(key=key, parent=parent, kv=kv)
                siblings[key] = node
                self.nodes[key] = node
                self.stats.inserted_tokens += node.tokens
            node.last_used = time.monotonic()
            return node

    def record(self, hit_tokens: int, miss_tokens: int) -> None:
        with self._lock:
            self.stats.hit_tokens += hit_tokens
            self.stats.miss_tokens += miss_tokens

    # eviction ---------------------------------------------------------------

    def enforce_budgets(self, protect: set[bytes] = frozenset()) -> None:
        with self._lock:
            gpu = self._bytes_on("cuda")
            if gpu > self.gpu_budget:
                for node in sorted(
                    (n for n in self.nodes.values() if n.kv.is_cuda and n.key not in protect),
                    key=lambda n: n.last_used,
                ):
                    if gpu <= self.gpu_budget:
                        break
                    if self.cpu_budget > 0:
                        host = torch.empty(node.kv.shape, dtype=node.kv.dtype, pin_memory=True)
                        host.copy_(node.kv)
                        node.kv = host
                        self.stats.offloads += 1
                        gpu -= node.nbytes
                    else:
                        gpu -= self._drop(node, count_gpu=True)
            cpu = self._bytes_on("cpu")
            if cpu > self.cpu_budget:
                for node in sorted(
                    (n for n in self.nodes.values() if not n.kv.is_cuda and n.key not in protect),
                    key=lambda n: n.last_used,
                ):
                    if cpu <= self.cpu_budget:
                        break
                    if node.key in self.nodes:
                        cpu -= self._drop(node, count_gpu=False)

    def _drop(self, node: Node, *, count_gpu: bool) -> int:
        """Remove ``node`` and its subtree; return the bytes freed on the counted tier."""
        freed = 0
        stack = [node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n.kv.is_cuda == count_gpu:
                freed += n.nbytes
            self.nodes.pop(n.key, None)
            self.stats.drops += 1
        siblings = node.parent.children if node.parent is not None else self.roots
        siblings.pop(node.key, None)
        return freed

    def clear(self) -> None:
        with self._lock:
            self.roots.clear()
            self.nodes.clear()
