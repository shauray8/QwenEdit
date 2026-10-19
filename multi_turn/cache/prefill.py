"""Prefill of prefix positions ``[start:]`` over cached K/V; exact because the prefix never attends forward."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as q21


def unwrap(model):
    return getattr(model, "_orig_mod", model)


def kv_geometry(model) -> tuple[int, int, int]:
    attn = model.transformer_blocks[0].attn
    return len(model.transformer_blocks), attn.heads, attn.head_dim


@torch.no_grad()
def prefix_inputs(model, encoder_hidden_states, layout, condition_latents, sample):
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
    total = prefix.shape[1]
    if start >= total:
        return
    zero_temb = model.time_text_embed(
        torch.zeros(1, device=prefix.device, dtype=prefix.dtype), prefix.dtype
    )
    scale1, gate1, scale2, gate2 = model.prepare_modulation(zero_temb)
    rope = layout["prefix_rope"][start:]
    segments = [(max(s, start), e, img) for s, e, img in layout["segments"] if e > start]
    for (s, _, is_image), (s0, _, _) in zip(segments, [g for g in layout["segments"] if g[1] > start]):
        if is_image and s0 < start:
            raise ValueError("cached prefix ends inside an image block")
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
            break
        out = attn.to_out[0](q21.cat_outputs(outputs, dim=1).flatten(2))[0]
        x = q21.residual_gate_add(x, out, gate1)
        x = q21.residual_gate_add(
            x, block.img_mlp(q21.apply_modulation(x, block.img_norm2, scale2)), gate2
        )
