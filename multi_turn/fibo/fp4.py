"""NVFP4 W4A4 for the Fibo DiT's self-attention and FFN linears (MULTI_TURN_NVFP4=1).

QKV and w1/w3 share their input, so each pair runs as one activation quantization and packed GEMMs. The Perceiver
and the injection cross-attentions stay BF16.

The activations of three of a block's four GEMMs are quantized inside the op that makes them, so the BF16 tensor
never round-trips through memory: RMSNorm + adaLN scale -> QKV and -> w1/w3, and SiLU * mul -> w2. The Triton
kernels write e2m1 pairs and e4m3 block scales (one per 16 values, times the layer's global scale) straight into the
128x4-swizzled layout ``flashinfer.mm_fp4`` reads, as ``flashinfer.fp4_quantize`` would.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import nn

from multi_turn.fibo.dit import _norm_scale, silu_mul
from multi_turn.pipeline.fp4 import ENABLED, NVFP4Linear

logger = logging.getLogger(__name__)
_converted: set[int] = set()


class Quantized(NamedTuple):
    """An NVFP4 activation: ``q`` [M, K/2] e2m1 pairs, ``sf`` swizzled e4m3 scales, ``shape`` of the BF16 tensor."""

    q: torch.Tensor
    sf: torch.Tensor
    shape: torch.Size


@triton.jit
def _e2m1(v):
    # round to nearest e2m1 {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even code, saturating; sign in bit 3
    a = tl.abs(v)
    code = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
            + (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return code | ((v < 0).to(tl.int32) << 3)


@triton.jit
def _quantize_store(y, row, col0, gs, q_ptr, sf_ptr, M, K: tl.constexpr, BLOCK: tl.constexpr):
    """Quantize ``y`` [BLOCK] fp32 (columns col0.. of ``row``) and store its e2m1 pairs and swizzled scales."""
    G: tl.constexpr = BLOCK // 16
    groups = tl.reshape(y, (G, 16))
    amax = tl.max(tl.abs(groups), axis=1)
    sf = (gs * (amax * (1.0 / 6.0))).to(tl.float8e4nv)
    sff = sf.to(tl.float32)
    scale = tl.where(sff != 0, 1.0 / (sff * (1.0 / gs)), 0.0)
    lo, hi = tl.split(tl.reshape(_e2m1(groups * scale[:, None]), (BLOCK // 2, 2)))
    cols = col0 // 2 + tl.arange(0, BLOCK // 2)
    tl.store(q_ptr + row * (K // 2) + cols, (lo | (hi << 4)).to(tl.uint8), mask=(cols < K // 2) & (row < M))
    # 128x4 tiles of scales: rows (m % 32, m % 128 // 32), then 4 consecutive groups
    g = col0 // 16 + tl.arange(0, G)
    offset = (row // 128) * (K // 64 * 512) + (g // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + g % 4
    tl.store(sf_ptr + offset, sf.to(tl.uint8, bitcast=True), mask=g < K // 16)


@triton.jit
def _norm_scale_fp4_kernel(x_ptr, w_ptr, noisy_ptr, clean_ptr, mask_ptr, gs_ptr, q_ptr, sf_ptr, M, S, mod_stride, eps,
                           K: tl.constexpr, BLOCK: tl.constexpr, HAS_MASK: tl.constexpr):
    row = tl.program_id(0)  # rows past M (to the 128-row tile) write zero scales
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * K + cols, mask=(cols < K) & (row < M), other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / K + eps)
    w = tl.load(w_ptr + cols, mask=cols < K, other=0.0).to(tl.float32)
    # BF16 roundings where the eager RMSNorm and modulation round
    y = ((x * rstd).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
    mod_row = (row // S) * mod_stride + cols  # adaLN chunks are column slices of one [B, 4K] output
    mod = tl.load(noisy_ptr + mod_row, mask=cols < K, other=0.0).to(tl.float32)
    if HAS_MASK:
        clean = tl.load(clean_ptr + mod_row, mask=cols < K, other=0.0).to(tl.float32)
        mod = tl.where(tl.load(mask_ptr + row % S) != 0, mod, clean)
    y = (y * (1.0 + mod).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    _quantize_store(y, row, 0, tl.load(gs_ptr), q_ptr, sf_ptr, M, K, BLOCK)


@triton.jit
def _silu_mul_fp4_kernel(gate_ptr, value_ptr, gs_ptr, q_ptr, sf_ptr, M, K: tl.constexpr, BLOCK: tl.constexpr):
    row, col0 = tl.program_id(0), tl.program_id(1) * BLOCK
    cols = col0 + tl.arange(0, BLOCK)
    valid = (cols < K) & (row < M)
    gate = tl.load(gate_ptr + row * K + cols, mask=valid, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + row * K + cols, mask=valid, other=0.0).to(tl.float32)
    y = ((gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32) * value).to(tl.bfloat16).to(tl.float32)
    _quantize_store(y, row, col0, tl.load(gs_ptr), q_ptr, sf_ptr, M, K, BLOCK)


def _buffers(x, k):
    m = x.numel() // k
    padded = -(-m // 128) * 128
    return m, padded, torch.empty(m, k // 2, dtype=torch.uint8, device=x.device), \
        torch.empty(padded, k // 16, dtype=torch.uint8, device=x.device)


def norm_scale_fp4(x, norm, noisy, clean, noise_mask, gs) -> Quantized:
    """``_norm_scale`` (RMSNorm, then the per-token adaLN scale) quantized to NVFP4 with global scale ``gs``."""
    k = x.shape[-1]
    m, padded, q, sf = _buffers(x, k)
    mask = noise_mask.view(torch.uint8) if noise_mask is not None else q
    _norm_scale_fp4_kernel[(padded,)](x, norm.weight, noisy, clean, mask, gs, q, sf, m, x.shape[-2], noisy.stride(0),
                                      norm.eps, K=k, BLOCK=triton.next_power_of_2(k), HAS_MASK=noise_mask is not None,
                                      num_warps=8)
    return Quantized(q, sf, x.shape)


def silu_mul_fp4(gate, value, gs, block=1024) -> Quantized:
    k = gate.shape[-1]
    m, padded, q, sf = _buffers(gate, k)
    _silu_mul_fp4_kernel[(padded, triton.cdiv(k, block))](gate, value, gs, q, sf, m, K=k, BLOCK=block, num_warps=4)
    return Quantized(q, sf, gate.shape)


class _Linear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.fp4 = NVFP4Linear([layer.weight], layer.bias)

    def forward(self, x):
        return self.fp4(x)[0]


def _quantized_input(norm, packed):
    def block_input(x, mods, noise_mask):
        if packed.act.scale is None or not x.is_contiguous():
            # the first call sets the layer's activation scale from the BF16 values
            return _norm_scale(x, norm.weight, norm.eps, *mods, noise_mask)
        return norm_scale_fp4(x, norm, *mods, noise_mask, packed.act.scale)

    return block_input


def _outputs(packed, x):
    return packed.quantized_outputs(*x) if isinstance(x, Quantized) else packed.outputs(x)


def _project_qkv(attn, packed):
    def project_qkv(x, context):
        shape = (*x.shape[:-1], attn.heads, attn.head_dim)
        q, k, v = _outputs(packed, x)
        return q.view(shape), k.view(shape), v.view(shape)

    return project_qkv


def _ffn_forward(packed, w2):
    def forward(x):
        gate, value = _outputs(packed, x)
        out = w2.fp4
        if out.act.scale is None:
            return w2(silu_mul(gate, value))
        y = out.quantized_outputs(*silu_mul_fp4(gate, value, out.act.scale))[0]
        return y if out.bias is None else y + out.bias

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
        block.attn_input = _quantized_input(block.attention_norm1, attn.qkv_fp4)
        block.ffn_input = _quantized_input(block.ffn_norm1, ffn.gate_up_fp4)
        del attn.to_q, attn.to_k, attn.to_v, ffn.w1, ffn.w3
    torch.cuda.empty_cache()
    logger.info("multi_turn: NVFP4 W4A4 Fibo DiT: packed QKV and w1/w3, %d blocks, %.1f -> %.1f GiB allocated",
                len(blocks), before / 2**30, torch.cuda.memory_allocated() / 2**30)
