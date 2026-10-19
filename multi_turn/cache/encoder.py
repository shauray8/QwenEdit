from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_rope import apply_qwen_vl_text_rope
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image21 as q21_stages,
)

from multi_turn import state
from multi_turn.cache import keys

logger = logging.getLogger(__name__)


def _now() -> float:
    if state.DEBUG and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _vision_inputs(images):
    out = []
    for image in images:
        if image.mode == "RGBA":
            white = Image.new("RGB", image.size, (255, 255, 255))
            white.paste(image, mask=image.getchannel("A"))
            image = white
        out.append(image)
    return out


def _prompt_text(prompt, num_images):
    prefix = " ".join(
        f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(num_images)
    )
    return (
        q21_stages.SYSTEM_TEMPLATE
        + f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    )


def _image_runs(ids: torch.Tensor, image_token_id: int, image_keys):
    mask = (ids == image_token_id).tolist()
    runs, start = [], None
    for position, is_pad in enumerate(mask + [False]):
        if is_pad and start is None:
            start = position
        elif not is_pad and start is not None:
            runs.append((start, position))
            start = None
    if len(runs) != len(image_keys):
        raise ValueError("image pad runs do not match the reference images")
    return [(s, e, k) for (s, e), k in zip(runs, image_keys)]


def _encoder_geometry(encoder):
    lm = encoder.model.language_model
    attn = lm.layers[0].self_attn
    return len(lm.layers), attn.num_key_value_heads, attn.head_dim, lm.config.hidden_size


def _cache_layers(past):
    if hasattr(past, "layers"):
        return [(layer.keys, layer.values) for layer in past.layers]
    return [past[i] for i in range(len(past))]


@torch.no_grad()
def encode_prompt_cached(stage, prompt, images, device, image_keys):
    encoder = stage.text_encoder
    marks = [("start", _now())]
    text = _prompt_text(prompt, len(images))
    kwargs = dict(text=[text], padding=True, padding_side="left", return_tensors="pt")
    if images:
        kwargs["images"] = _vision_inputs(images)
    inputs = stage.processor(**kwargs).to(device)
    marks.append(("processor", _now()))
    ids = inputs.input_ids[0]
    if not bool(inputs.attention_mask.all()):
        raise ValueError("unexpected padding in a single-prompt batch")

    runs = _image_runs(ids, stage.image_token_id, image_keys)
    vids = keys.sequence_ids(state.encoder_root(stage), ids.tolist(), runs)
    marks.append(("keys", _now()))

    with stage.use_declared_component(component_name="text_encoder", module=encoder) as enc:
        pool = state.encoder_pool(enc, device)
        match = pool.match(vids)
        marks.append(("match", _now()))
        try:
            hidden = _encode(stage, enc, inputs, ids, runs, vids, match, pool)
        finally:
            match.release()
        marks.append(("encode", _now()))
    if state.DEBUG:
        logger.info(
            "multi_turn: encoder reused %d/%d tokens; %s", match.length, len(vids),
            ", ".join(f"{name} {(t - marks[i][1]) * 1e3:.1f} ms" for i, (name, t) in enumerate(marks[1:])),
        )

    hidden = hidden[stage.drop_idx :]
    return q21_stages.collapse_image_slots(hidden, ids[stage.drop_idx :], stage.image_token_id)


def _encode(stage, enc, inputs, ids, runs, vids, match, pool):
    total, m = ids.shape[0], match.length
    if m == total:
        return pool.extra.index_select(0, match.indices)

    if m < max(state.ENC_MIN_REUSE, total // 4):  # a miss, or only the system header matched
        m = 0
        outputs = enc(**inputs, output_hidden_states=True, use_cache=True, logits_to_keep=1)
        hidden = outputs.hidden_states[-1][0]
        layers = _cache_layers(outputs.past_key_values)
        new_k = torch.stack([k[0].transpose(0, 1) for k, _ in layers])
        new_v = torch.stack([v[0].transpose(0, 1) for _, v in layers])
        _store(pool, vids, None, torch.stack([new_k, new_v], 1), hidden, 0)
        return hidden

    hidden_new, kv_new = _forward_tail(enc, inputs, ids, runs, match, pool)
    _store(pool, vids, match, kv_new, hidden_new, m)
    return torch.cat([pool.extra.index_select(0, match.indices), hidden_new])


def _store(pool, vids, match, kv_new, hidden_new, m):
    slots = pool.alloc(kv_new.shape[2])
    if slots is None:
        return
    pool.write(slots, kv_new)
    pool.extra.index_copy_(0, slots, hidden_new.to(pool.extra.dtype))
    indices = torch.cat([match.indices, slots]) if m else slots
    pool.insert(vids, indices, owned_from=m)


def _forward_tail(enc, inputs, ids, runs, match, pool):
    model = enc.model
    lm = model.language_model
    m, total = match.length, ids.shape[0]
    tail = ids[m:]
    hidden = lm.embed_tokens(tail[None])

    visual_mask = torch.zeros(total - m, dtype=torch.bool, device=hidden.device)
    deepstack_rows: list[list[torch.Tensor]] = []
    grid = inputs.image_grid_thw if "image_grid_thw" in inputs else None
    pixel_offsets = [0]
    if grid is not None:
        for row in grid.tolist():
            pixel_offsets.append(pixel_offsets[-1] + row[0] * row[1] * row[2])
    for index, (start, end, _) in enumerate(runs):
        if end <= m:
            continue
        pixels = inputs.pixel_values[pixel_offsets[index] : pixel_offsets[index + 1]]
        embeds, deepstack = model._get_flat_visual_features(pixels, grid[index : index + 1])
        first = max(start, m) - start
        lo, hi = max(start, m) - m, end - m
        hidden[0, lo:hi] = embeds[first:].to(hidden.dtype)
        visual_mask[lo:hi] = True
        for layer, rows in enumerate(deepstack):
            if layer == len(deepstack_rows):
                deepstack_rows.append([])
            deepstack_rows[layer].append(rows[first:])
    deepstack_embeds = [torch.cat(rows) for rows in deepstack_rows]

    position_ids, _ = model.get_rope_index(
        ids[None], grid, None, attention_mask=inputs.attention_mask
    )
    position_ids = position_ids[:, :, m:]
    cached = pool.gather(match.indices)
    n = total - m
    # bottom-right causal: tail position i sees cached [0, m) and tail [0, i]
    mask = torch.ones(n, total, dtype=torch.bool, device=hidden.device).tril(diagonal=m)
    kv_new = torch.empty(len(lm.layers), 2, n, *cached.shape[3:], dtype=cached.dtype, device=cached.device)

    for index, layer in enumerate(lm.layers):
        attn = layer.self_attn
        residual = hidden
        x = layer.input_layernorm(hidden)
        shape = (1, n, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_qwen_vl_text_rope(attn.rotary_emb, position_ids, q, k)
        kv_new[index, 0] = k[0].transpose(0, 1)
        kv_new[index, 1] = v[0].transpose(0, 1)
        keys_ = torch.cat([cached[index, 0].transpose(0, 1)[None], k], dim=2)
        values = torch.cat([cached[index, 1].transpose(0, 1)[None], v], dim=2)
        out = F.scaled_dot_product_attention(
            q, keys_, values, attn_mask=mask, scale=attn.scaling, enable_gqa=True
        )
        out = attn.o_proj(out.transpose(1, 2).reshape(1, n, -1))
        hidden = residual + out
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        if index < len(deepstack_embeds) and deepstack_embeds:
            hidden[0, visual_mask] += deepstack_embeds[index].to(hidden.dtype)

    return hidden[0], kv_new
