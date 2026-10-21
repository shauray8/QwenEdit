"""NVFP4 W4A4 for the Fibo DiT's self-attention and FFN linears (MULTI_TURN_NVFP4=1).

QKV and w1/w3 share their input, so each pair runs as one activation quantization and packed GEMMs. The Perceiver
and the injection cross-attentions stay BF16.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from multi_turn.pipeline.fp4 import ENABLED, NVFP4Linear

logger = logging.getLogger(__name__)
_converted: set[int] = set()


class _Linear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.fp4 = NVFP4Linear([layer.weight], layer.bias)

    def forward(self, x):
        return self.fp4(x)[0]


def _project_qkv(attn, packed):
    def project_qkv(x, context):
        shape = (*x.shape[:-1], attn.heads, attn.head_dim)
        q, k, v = packed.outputs(x)
        return q.view(shape), k.view(shape), v.view(shape)

    return project_qkv


def _ffn_forward(packed, w2):
    def forward(x):
        gate, value = packed.outputs(x)
        return w2(F.silu(gate) * value)

    return forward


@torch.no_grad()
def maybe_convert(model: nn.Module) -> None:
    if not ENABLED or id(model) in _converted or hasattr(model.layers[0].attention, "qkv_fp4"):
        return
    _converted.add(id(model))
    before = torch.cuda.memory_allocated()
    blocks = list(model.noise_refiner) + list(model.layers)
    for block in blocks:
        attn, ffn = block.attention, block.feed_forward
        attn.qkv_fp4 = NVFP4Linear([attn.to_q.weight, attn.to_k.weight, attn.to_v.weight])
        attn.project_qkv = _project_qkv(attn, attn.qkv_fp4)
        attn.to_out[0] = _Linear(attn.to_out[0])
        ffn.gate_up_fp4 = NVFP4Linear([ffn.w1.weight, ffn.w3.weight])
        ffn.w2 = _Linear(ffn.w2)
        ffn.forward = _ffn_forward(ffn.gate_up_fp4, ffn.w2)
        del attn.to_q, attn.to_k, attn.to_v, ffn.w1, ffn.w3
    torch.cuda.empty_cache()
    logger.info("multi_turn: NVFP4 W4A4 Fibo DiT: packed QKV and w1/w3, %d blocks, %.1f -> %.1f GiB allocated",
                len(blocks), before / 2**30, torch.cuda.memory_allocated() / 2**30)
