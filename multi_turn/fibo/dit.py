"""Fibo 2.0 DiT for SGLang-Diffusion: a port of diffusers' ``BriaFibo2Transformer2DModel``.

Parameter names match the checkpoint. Self-attention runs through SGLang's ``USPAttention`` (Sage, FA, SDPA);
the Perceiver and the injection cross-attentions are small and use SDPA. The elementwise glue (RMSNorm, adaLN
modulation, gates, RoPE) is ~60% of a step in eager mode, so it runs through ``torch.compile``d helpers
(MULTI_TURN_FIBO_COMPILE=0 keeps it eager).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT

ADALN_EMBED_DIM = 256
COMPILE = os.environ.get("MULTI_TURN_FIBO_COMPILE", "1") == "1"


def _compiled(fn):
    return torch.compile(fn, dynamic=True) if COMPILE else fn


def _rms_norm_eager(x, weight, eps):
    # rounds to the weight's dtype before the scale, as diffusers' RMSNorm does
    x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return x.to(weight.dtype) * weight


def _mod(noisy, clean, noise_mask):
    # per-token adaLN values: the timestep's on the tokens being generated, t=0's on the clean context tokens
    if noise_mask is None:
        return noisy[:, None]
    return torch.where(noise_mask[None, :, None], noisy[:, None], clean[:, None])


_rms_norm = _compiled(_rms_norm_eager)


@_compiled
def _norm_scale(x, weight, eps, noisy, clean, noise_mask):
    return _rms_norm_eager(x, weight, eps) * (1 + _mod(noisy, clean, noise_mask))


@_compiled
def _gated_residual(x, y, weight, eps, noisy, clean, noise_mask):
    return x + _mod(noisy, clean, noise_mask).tanh() * _rms_norm_eager(y, weight, eps)


@dataclass
class FiboArchConfig(DiTArchConfig):
    in_channels: int = 32
    patch_size: int = 2
    dim: int = 3584
    n_layers: int = 44
    n_refiner_layers: int = 2
    n_heads: int = 28
    norm_eps: float = 1e-5
    cap_feat_dim: int = 7680
    injection_layer_ids: tuple[int, ...] = (6, 14, 22, 30, 38)
    perceiver_num_layers: int = 2
    min_num_gist_tokens: int = 256
    max_num_gist_tokens: int = 512
    gist_step: int = 64
    gist_min_text_len: int = 256
    gist_max_text_len: int = 2048
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (2048, 512, 512)
    lora_param_names_mapping: dict = field(default_factory=lambda: {r"^transformer\.": ""})

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.dim
        self.num_attention_heads = self.n_heads
        # the pipeline packs 2x2 latent patches, so a token carries 4x the VAE channels
        self.num_channels_latents = self.in_channels * self.patch_size**2


@dataclass
class FiboDitConfig(DiTConfig):
    arch_config: FiboArchConfig = field(default_factory=FiboArchConfig)
    prefix: str = "fibo2"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return _rms_norm(x, self.weight, self.eps)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, mid_size), nn.SiLU(), nn.Linear(mid_size, out_size))
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(silu_mul(self.w1(x), self.w3(x)))


@_compiled
def silu_mul(gate, value):
    return F.silu(gate) * value


def _rope_eager(x, freqs_cis):
    out = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs_cis.unsqueeze(2)
    return torch.view_as_real(out).flatten(3).type_as(x)


@_compiled
def _norm_rope(x, weight, eps, freqs_cis):
    x = _rms_norm_eager(x, weight, eps)
    return x if freqs_cis is None else _rope_eager(x, freqs_cis)


class Attention(nn.Module):
    """Self-attention with RoPE (``usp``), or SDPA cross-attention to ``context``."""

    def __init__(self, dim, heads, kv_dim=None, eps=1e-5, usp=True):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        kv_dim = kv_dim or dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.norm_q = RMSNorm(self.head_dim, eps)
        self.norm_k = RMSNorm(self.head_dim, eps)
        self.attn = USPAttention(num_heads=heads, head_size=self.head_dim, causal=False) if usp else None

    def project_qkv(self, x, context):
        shape = (*x.shape[:-1], self.heads, self.head_dim)
        kv_shape = (*context.shape[:-1], self.heads, self.head_dim)
        return self.to_q(x).view(shape), self.to_k(context).view(kv_shape), self.to_v(context).view(kv_shape)

    def context_kv(self, context):
        """Normalized K and V of a cross-attention context, reusable while the context stays the same."""
        kv_shape = (*context.shape[:-1], self.heads, self.head_dim)
        return self.norm_k(self.to_k(context).view(kv_shape)), self.to_v(context).view(kv_shape)

    def forward(self, x, context=None, mask=None, freqs_cis=None, context_kv=None):
        if context_kv is None:
            q, k, v = self.project_qkv(x, x if context is None else context)
            k = _norm_rope(k, self.norm_k.weight, self.norm_k.eps, freqs_cis)
        else:
            q, (k, v) = self.to_q(x).view(*x.shape[:-1], self.heads, self.head_dim), context_kv
        q = _norm_rope(q, self.norm_q.weight, self.norm_q.eps, freqs_cis)
        if self.attn is not None and mask is None:
            out = self.attn(q, k, v)
        else:
            m = None if mask is None else mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=m)
            out = out.transpose(1, 2)
        return self.to_out[0](out.flatten(2))



class Block(nn.Module):
    def __init__(self, dim, n_heads, eps, cross_dim=None):
        super().__init__()
        self.attention = Attention(dim, n_heads, eps=eps)
        self.feed_forward = FeedForward(dim, (dim * 8 // 3) // 64 * 64)
        self.attention_norm1 = RMSNorm(dim, eps)
        self.ffn_norm1 = RMSNorm(dim, eps)
        self.attention_norm2 = RMSNorm(dim, eps)
        self.ffn_norm2 = RMSNorm(dim, eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))
        self.cross_attention = None
        if cross_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps)
            self.cross_attention = Attention(dim, n_heads, kv_dim=cross_dim, eps=eps, usp=False)
            self.cross_attn_norm2 = RMSNorm(dim, eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def modulation(self, adaln_noisy, adaln_clean):
        """``(scale_msa, gate_msa, scale_mlp, gate_mlp)`` as (noisy, clean) pairs, selected per token later."""
        noisy = self.adaLN_modulation(adaln_noisy).chunk(4, dim=1)
        clean = self.adaLN_modulation(adaln_clean).chunk(4, dim=1) if adaln_clean is not None else noisy
        return list(zip(noisy, clean))

    def forward(self, x, freqs_cis, adaln_noisy, adaln_clean, noise_mask, mask=None, text_kv=None):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.modulation(adaln_noisy,
                                                                   None if noise_mask is None else adaln_clean)
        n1, n2 = self.attention_norm1, self.attention_norm2
        attn = self.attention(_norm_scale(x, n1.weight, n1.eps, *scale_msa, noise_mask), mask=mask, freqs_cis=freqs_cis)
        x = _gated_residual(x, attn, n2.weight, n2.eps, *gate_msa, noise_mask)
        if self.cross_attention is not None:
            cross = self.cross_attention(self.cross_attn_norm1(x), context_kv=text_kv)
            gate = self.cross_attn_gate.view(1, 1)
            x = _gated_residual(x, cross, self.cross_attn_norm2.weight, self.cross_attn_norm2.eps, gate, gate, None)
        f1, f2 = self.ffn_norm1, self.ffn_norm2
        ffn = self.feed_forward(_norm_scale(x, f1.weight, f1.eps, *scale_mlp, noise_mask))
        return _gated_residual(x, ffn, f2.weight, f2.eps, *gate_mlp, noise_mask)


class PerceiverLayer(nn.Module):
    def __init__(self, dim, n_heads, cross_dim, eps):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps)
        self.cross_attention = Attention(dim, n_heads, kv_dim=cross_dim, eps=eps, usp=False)
        self.self_attn_norm = RMSNorm(dim, eps)
        self.self_attention = Attention(dim, n_heads, eps=eps, usp=False)
        self.ffn_norm = RMSNorm(dim, eps)
        self.ffn = FeedForward(dim, (dim * 8 // 3) // 64 * 64)

    def forward(self, latents, text, text_mask=None, mask=None):
        latents = latents + self.cross_attention(self.cross_attn_norm(latents), context=text, mask=text_mask)
        latents = latents + self.self_attention(self.self_attn_norm(latents), mask=mask)
        return latents + self.ffn(self.ffn_norm(latents))


class Perceiver(nn.Module):
    def __init__(self, dim, n_heads, cross_dim, num_queries, num_layers, eps):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([PerceiverLayer(dim, n_heads, cross_dim, eps) for _ in range(num_layers)])
        self.norm_out = RMSNorm(dim, eps)

    def forward(self, text, n, text_mask=None):
        latents = self.latent_queries[:n].unsqueeze(0).expand(text.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(latents, text, text_mask)
        return self.norm_out(latents)


class FinalLayer(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(dim, ADALN_EMBED_DIM), dim, bias=True))

    def forward(self, x, c):
        return self.linear(self.norm_final(x) * (1 + self.adaLN_modulation(c))[:, None])


class RopeEmbedder:
    def __init__(self, theta, axes_dims, axes_lens):
        self.theta, self.axes_dims, self.axes_lens = theta, axes_dims, axes_lens
        self.freqs = None

    def __call__(self, ids):
        if self.freqs is None or self.freqs[0].device != ids.device:
            self.freqs = []
            for d, e in zip(self.axes_dims, self.axes_lens):
                f = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
                f = torch.outer(torch.arange(e, dtype=torch.float64), f).float()
                self.freqs.append(torch.polar(torch.ones_like(f), f).to(torch.complex64).to(ids.device))
        return torch.cat([self.freqs[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def _grid(size, start, device):
    axes = [torch.arange(s, s + n, dtype=torch.int32, device=device) for s, n in zip(start, size)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 2)


def gist_count(text_len, c: FiboArchConfig) -> int:
    ratio = min(1.0, max(0.0, (text_len - c.gist_min_text_len) / (c.gist_max_text_len - c.gist_min_text_len)))
    raw = c.min_num_gist_tokens + ratio * (c.max_num_gist_tokens - c.min_num_gist_tokens)
    return max(c.min_num_gist_tokens, min(c.max_num_gist_tokens, math.ceil(raw / c.gist_step) * c.gist_step))


class FiboTransformer2DModel(CachableDiT):
    _fsdp_shard_conditions = []
    _compile_conditions = []
    param_names_mapping = {}
    reverse_param_names_mapping = {}

    def __init__(self, config: FiboDitConfig, hf_config: dict, quant_config=None, **kwargs):
        super().__init__(config=config, hf_config=hf_config)
        c = self.config
        self.hidden_size, self.num_attention_heads = c.dim, c.n_heads
        self.num_channels_latents = c.num_channels_latents
        token_dim = c.patch_size**2 * c.in_channels
        self.x_embedder = nn.Linear(token_dim, c.dim, bias=True)
        self.t_embedder = TimestepEmbedder(min(c.dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = RopeEmbedder(c.rope_theta, c.axes_dims, c.axes_lens)
        self.noise_refiner = nn.ModuleList([Block(c.dim, c.n_heads, c.norm_eps) for _ in range(c.n_refiner_layers)])
        self.perceiver = Perceiver(c.dim, c.n_heads, c.cap_feat_dim, c.max_num_gist_tokens, c.perceiver_num_layers,
                                   c.norm_eps)
        self.layers = nn.ModuleList([
            Block(c.dim, c.n_heads, c.norm_eps, cross_dim=c.cap_feat_dim if i in c.injection_layer_ids else None)
            for i in range(c.n_layers)
        ])
        self.final_layer = FinalLayer(c.dim, token_dim)
        self._bundle = {layer: i + 1 for i, layer in enumerate(c.injection_layer_ids)}

    def forward(self, hidden_states, encoder_hidden_states, timestep, latent_hw=None, context_tokens=None,
                context_hw=None, guidance=None, text_cache=None, **kwargs):
        """``hidden_states`` [B, N, 128] packed latent tokens at ``latent_hw`` (in tokens); ``encoder_hidden_states``
        [B, 6, T, 7680] text bundles; ``context_tokens`` [B, Nc, 128] for the images to edit, laid out at
        ``context_hw``; ``timestep`` is SGLang's sigma * 1000. ``text_cache``, a dict kept across a request's steps,
        holds the Perceiver output and the injection blocks' K/V, which depend only on the text."""
        c = self.config
        if isinstance(encoder_hidden_states, (list, tuple)):
            encoder_hidden_states = encoder_hidden_states[0]
        device, dtype = hidden_states.device, self.x_embedder.weight.dtype
        b, n_image = hidden_states.shape[:2]
        h, w = latent_hw
        t = timestep.reshape(-1)[:1].expand(b).float() / 1000.0
        adaln_noisy = self.t_embedder(t * c.t_scale)
        adaln_clean = self.t_embedder(torch.zeros_like(t))

        plane = c.max_num_gist_tokens + 1
        tokens, ids = [hidden_states.to(dtype)], [_grid((1, h, w), (plane, 0, 0), device)]
        for k, (ch, cw) in enumerate(context_hw or []):
            ids.append(_grid((1, ch, cw), (plane + k + 1, 0, 0), device))
        if context_tokens is not None:
            tokens.append(context_tokens.to(dtype))
        x = self.x_embedder(torch.cat(tokens, dim=1))
        image_freqs = self.rope_embedder(torch.cat(ids)).unsqueeze(0)
        n_total = x.shape[1]
        noise_mask = None
        if n_total > n_image:
            noise_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
            noise_mask[:n_image] = True
        for block in self.noise_refiner:
            x = block(x, image_freqs, adaln_noisy, adaln_clean, noise_mask)

        text = encoder_hidden_states.to(dtype)
        n_gist = gist_count(text.shape[2], c)
        cache = {} if text_cache is None else text_cache
        if "gist" not in cache:
            cache["gist"] = self.perceiver(text[:, 0], n_gist)
            cache["kv"] = {i: self.layers[i].cross_attention.context_kv(text[:, bundle])
                           for i, bundle in self._bundle.items()}
        gist = cache["gist"]
        gist_freqs = self.rope_embedder(_grid((n_gist, 1, 1), (1, 0, 0), device)).unsqueeze(0)
        x = torch.cat([gist, x], dim=1)
        freqs = torch.cat([gist_freqs, image_freqs], dim=1)
        noise_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
        noise_mask[n_gist:n_gist + n_image] = True
        for i, block in enumerate(self.layers):
            x = block(x, freqs, adaln_noisy, adaln_clean, noise_mask, text_kv=cache["kv"].get(i))
        return self.final_layer(x[:, n_gist:n_gist + n_image], adaln_noisy).float()


EntryClass = FiboTransformer2DModel
