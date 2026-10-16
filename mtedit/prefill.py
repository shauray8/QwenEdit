"""Prefix prefill that starts from cached K/V.

Mirrors the prefix branch of ``QwenImage21Attention.attend_sample`` and the
prefix update in ``QwenImage21TransformerBlock.forward`` (sglang
``runtime/models/dits/qwen_image21.py``), but only for prefix positions
``[start:]``: earlier positions contribute their cached K/V and nothing else,
which is exact because nothing in the prefix attends forward.

The result is written into one ``[layers, 2, P, heads, head_dim]`` buffer whose
per-layer views become the request's ``prefix_caches``, so the DiT then runs
its normal cached path (and breakable-CUDA-graph replay) unchanged.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as q21


def unwrap(model):
    # torch.compile wraps the module; the submodules live on the original
    return getattr(model, "_orig_mod", model)


def kv_geometry(model) -> tuple[int, int, int]:
    attn = model.transformer_blocks[0].attn
    return len(model.transformer_blocks), attn.heads, attn.head_dim


@torch.no_grad()
def prefix_inputs(model, encoder_hidden_states, layout, condition_latents, sample):
    """The prefix hidden states exactly as ``QwenImage21Transformer2DModel.forward`` builds them."""
    prefix = model.txt_in(
        encoder_hidden_states[sample : sample + 1, : layout["encoder_seq_len"]]
    ).index_select(1, layout["text_indices"])
    if condition_latents is not None:
        prefix[:, layout["image_indices"]] = model.img_in(
            condition_latents[sample : sample + 1]
        )
    return prefix


@torch.no_grad()
def prefill_from(model, prefix, layout, kv: torch.Tensor, start: int) -> None:
    """Fill ``kv[:, :, start:]`` for one sample; ``kv[:, :, :start]`` already holds cached K/V.

    ``prefix`` is the full ``[1, P, C]`` prefix input; ``kv`` is
    ``[layers, 2, P, heads, head_dim]``.
    """
    total = prefix.shape[1]
    if start >= total:
        return
    zero_temb = model.time_text_embed(
        torch.zeros(1, device=prefix.device, dtype=prefix.dtype), prefix.dtype
    )
    scale1, gate1, scale2, gate2 = model.prepare_modulation(zero_temb)
    rope = layout["prefix_rope"][start:]
    segments = [(s, e, img) for s, e, img in layout["segments"] if e > start]
    if segments and segments[0][0] < start:
        raise ValueError("cached prefix must end on a segment boundary")
    x = prefix[:, start:]
    blocks = model.transformer_blocks
    for layer, block in enumerate(blocks):
        attn = block.attn
        q, k, v = attn.qkv(q21.apply_modulation(x, block.img_norm1, scale1), rope)
        kv[layer, 0, start:] = k[0]
        kv[layer, 1, start:] = v[0]
        keys, values = kv[layer, 0][None], kv[layer, 1][None]
        outputs = []
        for s, e, is_image in segments:
            mask = None
            if not is_image:
                mask = (
                    torch.arange(e, device=q.device)[None, :]
                    <= torch.arange(s, e, device=q.device)[:, None]
                )[None, None]
            outputs.append(
                attn.local_attn(
                    q[:, s - start : e - start], keys[:, :e], values[:, :e], attn_mask=mask
                )
            )
        if layer == len(blocks) - 1:
            break  # the last layer's prefix states feed nothing
        out = attn.to_out[0](q21.cat_outputs(outputs, dim=1).flatten(2))[0]
        x = q21.residual_gate_add(x, out, gate1)
        x = q21.residual_gate_add(
            x, block.img_mlp(q21.apply_modulation(x, block.img_norm2, scale2)), gate2
        )
