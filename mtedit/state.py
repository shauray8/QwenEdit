"""Worker-process singletons and env configuration.

MTEDIT_PREFIX_CACHE   1 (default) enables cross-request prefix K/V reuse for every Q21 request
MTEDIT_KV_GPU_GIB     GPU budget for stored prefix K/V (default 24)
MTEDIT_KV_CPU_GIB     pinned-host budget; 0 disables the host tier (default 64)
MTEDIT_LATENT_SLOTS   stored output latents / VAE encodes kept (default 512 each)
MTEDIT_DEBUG          1 logs per-request hit/miss details
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch

from mtedit.kvstore import GIB, PrefixKVStore


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


PREFIX_CACHE = os.environ.get("MTEDIT_PREFIX_CACHE", "1") == "1"
DEBUG = os.environ.get("MTEDIT_DEBUG", "0") == "1"


class LRU:
    """A small LRU of tensors (output latents, VAE encodes); a few MiB per entry."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: OrderedDict = OrderedDict()

    def get(self, key):
        value = self.items.get(key)
        if value is not None:
            self.items.move_to_end(key)
        return value

    def put(self, key, value) -> None:
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)


_slots = int(os.environ.get("MTEDIT_LATENT_SLOTS", 512))
STORE = PrefixKVStore(
    gpu_budget=int(_env_float("MTEDIT_KV_GPU_GIB", 24) * GIB),
    cpu_budget=int(_env_float("MTEDIT_KV_CPU_GIB", 64) * GIB),
)
# latent id -> [1, HW, C] final latents of a stored output (normalized, packed, on GPU)
OUTPUT_LATENTS = LRU(_slots)
# image content key -> [1, HW, C] VAE condition latents
VAE_LATENTS = LRU(_slots)


def reset() -> None:
    STORE.clear()
    OUTPUT_LATENTS.items.clear()
    VAE_LATENTS.items.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
