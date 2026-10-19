"""NVFP4 W4A4 DiT linears via ``flashinfer.mm_fp4``, swapped in on the first denoising call (MULTI_TURN_NVFP4=1)."""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MULTI_TURN_NVFP4", "0") == "1"
_FP4_MAX, _FP8_MAX = 6.0, 448.0
_converted: set[int] = set()
_BACKEND = os.environ.get("MULTI_TURN_NVFP4_BACKEND", "b12x")


def _global_scale(t: torch.Tensor) -> torch.Tensor:
    return (_FP8_MAX * _FP4_MAX / t.float().abs().amax().clamp(min=1e-8)).to(torch.float32)


class _ActQuant:

    HEADROOM = float(os.environ.get("MULTI_TURN_NVFP4_HEADROOM", 4.0))

    def __init__(self):
        self.scale = None

    def __call__(self, x2):
        import flashinfer

        if self.scale is None:
            amax = x2.float().abs().amax().clamp(min=1e-8) * self.HEADROOM
            self.scale = (_FP8_MAX * _FP4_MAX / amax).to(torch.float32)
        xq, xsf = flashinfer.fp4_quantize(x2, self.scale)
        return xq, xsf, self.scale


class NVFP4Weight:
    def __init__(self, weight: torch.Tensor):
        import flashinfer

        w = weight.detach().to(torch.bfloat16).contiguous()
        self.n = w.shape[0]
        self.gw = _global_scale(w)
        self.wq, self.wsf = flashinfer.fp4_quantize(w, self.gw)

    def mm(self, xq, xsf, gx):
        import flashinfer

        return flashinfer.mm_fp4(xq, self.wq.t(), xsf, self.wsf.t(), (1.0 / (gx * self.gw)).to(torch.float32),
                                 torch.bfloat16, backend=_BACKEND)


class NVFP4Linear(nn.Module):

    quant_method = None  # keeps the DiT's BF16-only fast paths off

    def __init__(self, weights, bias: torch.Tensor | None = None):
        super().__init__()
        self.weights = [NVFP4Weight(w) for w in weights]
        self.act = _ActQuant()
        self.bias = None if bias is None else bias.detach().to(torch.bfloat16)

    def outputs(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.dtype != torch.bfloat16 or not x2.is_contiguous():
            x2 = x2.to(torch.bfloat16).contiguous()
        xq, xsf, gx = self.act(x2)
        return [w.mm(xq, xsf, gx).view(*shape[:-1], w.n) for w in self.weights]

    def forward(self, x):
        out = self.outputs(x)[0]
        if self.bias is not None:
            out = out + self.bias
        return out, None


def _swap(parent: nn.Module, name: str) -> bool:
    layer = getattr(parent, name, None)
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return False
    setattr(parent, name, NVFP4Linear([weight], getattr(layer, "bias", None)))
    return True


class _Packed(nn.Module):

    quant_method = None
    weight = torch.empty(0)

    def forward(self, x):
        raise RuntimeError("this linear is folded into a packed NVFP4 GEMM")


def _pack(layers) -> NVFP4Linear:
    return NVFP4Linear([layer.weight for layer in layers])


def _attn_project_qkv(attn):
    def project_qkv(x):
        shape = (*x.shape[:-1], attn.heads, attn.head_dim)
        q, k, v = attn.qkv_fp4.outputs(x)
        return q.view(shape), k.view(shape), v.view(shape)

    return project_qkv


def _ffn_forward(mlp):
    from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as q21

    fused = getattr(q21, "fused_silu_mul_bitexact", None)

    def forward(x):
        gate, value = mlp.gate_up_fp4.outputs(x)
        hidden = fused(gate, value) if fused is not None else torch.nn.functional.silu(gate) * value
        return mlp.out(hidden)[0]

    return forward


@torch.no_grad()
def maybe_convert(model: nn.Module) -> None:
    if not ENABLED or id(model) in _converted:
        return
    _converted.add(id(model))
    for block in model.transformer_blocks:
        attn, mlp = block.attn, block.img_mlp
        attn.qkv_fp4 = _pack([attn.to_q, attn.to_k, attn.to_v])
        attn.project_qkv = _attn_project_qkv(attn)
        _swap(attn.to_out, "0")
        mlp.gate_up_fp4 = _pack([mlp.gate_layer, mlp.proj])
        mlp.forward = _ffn_forward(mlp)
        _swap(mlp, "out")
        for owner, name in ((attn, "to_q"), (attn, "to_k"), (attn, "to_v"), (mlp, "gate_layer"), (mlp, "proj")):
            setattr(owner, name, _Packed())
    torch.cuda.empty_cache()
    logger.info("multi_turn: NVFP4 W4A4 DiT: packed QKV and gate/up, %d blocks", len(model.transformer_blocks))
