"""Worker-process singletons and env configuration.

MTEDIT_PREFIX_CACHE    1 (default): DiT prefix K/V reuse across requests
MTEDIT_ENCODER_CACHE   1 (default): text-encoder (Qwen3-VL) prefix K/V reuse across requests
MTEDIT_KV_GPU_GIB      DiT pool size (default 24; 512 KiB/token, a 1024^2 reference is 4096 tokens)
MTEDIT_ENC_GPU_GIB     encoder pool size (default 4; ~152 KiB/token)
MTEDIT_MIN_REUSE       shortest DiT prefix match worth assembling (default 256 tokens)
MTEDIT_ENC_MIN_REUSE   shortest encoder match worth a tail-only forward (default 128, and >= 1/4 of it)
MTEDIT_LATENT_SLOTS    stored output latents / VAE encodes kept (default 512 each)
MTEDIT_DEBUG           1 logs per-request reuse
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch

from mtedit import keys

logger = logging.getLogger(__name__)
GIB = 1 << 30


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


PREFIX_CACHE = os.environ.get("MTEDIT_PREFIX_CACHE", "1") == "1"
ENCODER_CACHE = os.environ.get("MTEDIT_ENCODER_CACHE", "1") == "1"
DEBUG = os.environ.get("MTEDIT_DEBUG", "0") == "1"
MIN_REUSE = int(os.environ.get("MTEDIT_MIN_REUSE", 256))
# shortest text-encoder match worth a tail-only forward (also at least a quarter of the sequence)
ENC_MIN_REUSE = int(os.environ.get("MTEDIT_ENC_MIN_REUSE", 128))


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
# latent id -> [1, HW, C] final latents of a stored output (normalized, packed, on GPU)
OUTPUT_LATENTS = LRU(_slots)
# latent id -> (h, w) latent grid of the stored output
OUTPUT_GRID: dict[str, tuple[int, int]] = {}


def stored_latents(latent_id, crop=None):
    """A stored output's latents, optionally a (x0, y0, x1, y1) window in latent cells; None if absent."""
    latents = OUTPUT_LATENTS.get(latent_id) if latent_id else None
    if latents is None or crop is None:
        return latents
    grid = OUTPUT_GRID.get(latent_id)
    if grid is None:
        return None
    x0, y0, x1, y1 = crop
    h, w = grid
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        return None
    return latents.view(1, h, w, -1)[:, y0:y1, x0:x1].reshape(1, (y1 - y0) * (x1 - x0), -1)
# image content key -> [1, HW, C] VAE condition latents
VAE_LATENTS = LRU(_slots)

_dit_pool = None
_encoder_pool = None


def dit_pool(model):
    """The DiT prefix pool, sized on first use from the transformer's geometry."""
    global _dit_pool
    if _dit_pool is None:
        from mtedit.paged import PagedKVPool
        from mtedit.prefill import kv_geometry

        layers, heads, head_dim = kv_geometry(model)
        weight = model.img_in.weight
        _dit_pool = PagedKVPool(
            "dit", layers, heads, head_dim,
            int(_env_float("MTEDIT_KV_GPU_GIB", 24) * GIB), torch.bfloat16, weight.device,
        )
    return _dit_pool


def encoder_pool(encoder, device):
    """The text-encoder pool: per-layer K/V plus the last-layer hidden row per token."""
    global _encoder_pool
    if _encoder_pool is None:
        from mtedit.encoder_cache import _encoder_geometry
        from mtedit.paged import PagedKVPool

        layers, kv_heads, head_dim, hidden = _encoder_geometry(encoder)
        _encoder_pool = PagedKVPool(
            "encoder", layers, kv_heads, head_dim,
            int(_env_float("MTEDIT_ENC_GPU_GIB", 4) * GIB), torch.bfloat16, device, extra_dim=hidden,
        )
    return _encoder_pool


def _weights_epoch():
    try:
        from sglang.multimodal_gen.runtime.cache.conditioning import conditioning_weights_epoch

        return conditioning_weights_epoch()
    except Exception:
        return None


def encoder_root(stage) -> bytes:
    config = getattr(stage.text_encoder, "config", None)
    name = getattr(config, "_name_or_path", None) or type(stage.text_encoder).__name__
    return keys.root_key("encoder", name, _weights_epoch())


def dit_root(server_args) -> bytes:
    return keys.root_key(
        "dit",
        server_args.model_path,
        getattr(server_args, "component_quantizations", None),
        getattr(server_args, "transformer_quantization", None),
        _weights_epoch(),
    )


def usage() -> dict:
    return {
        "dit": _dit_pool.usage() if _dit_pool else None,
        "encoder": _encoder_pool.usage() if _encoder_pool else None,
        "output_latents": len(OUTPUT_LATENTS.items),
        "vae_latents": len(VAE_LATENTS.items),
    }


def reset() -> None:
    for pool in (_dit_pool, _encoder_pool):
        if pool is not None:
            pool.clear()
    OUTPUT_LATENTS.items.clear()
    VAE_LATENTS.items.clear()
