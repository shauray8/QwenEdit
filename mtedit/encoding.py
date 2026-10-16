"""Replacement for ``QwenImage21EncodingStage.forward``.

A copy of the upstream method (identical at sglang e6072935b and on main as of
2026-10-17) with three additions, each marked ``# mt:``:

1. condition latents come from a stored output (latent passthrough) or the
   VAE-latent cache before falling back to a VAE encode;
2. per-sample segment chain keys are computed for the prefix K/V store;
3. each branch's ``prefix_caches`` list is tagged with those keys (``MTCaches``)
   so the denoising hook can find them without a new kwarg.

It also prepares the latent-mask state for region edits.
"""

from __future__ import annotations

import logging
import math

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image21 as q21_stages,
)
from sglang.multimodal_gen.runtime.utils.vision import load_image

from mtedit import keys, state
from mtedit.masking import MaskState, rasterize
from mtedit.request import MTRequest

logger = logging.getLogger(__name__)

MASK_EXTRA_KEY = "mt_mask"


class MTCaches(list):
    """``prefix_caches`` for one branch, carrying the per-sample segment chains."""

    def __init__(self, caches, chains, prefix_lengths):
        super().__init__(caches)
        self.chains = chains  # per sample: list[(start, end, key)] or None
        self.prefix_lengths = prefix_lengths
        self.prepared = False
        self.reused_from: list[int] = [0] * len(caches)  # per sample: first segment not from the store


def _root_key(server_args) -> bytes:
    try:
        from sglang.multimodal_gen.runtime.cache.conditioning import conditioning_weights_epoch

        epoch = conditioning_weights_epoch()
    except Exception:  # older/newer sglang without the epoch counter
        epoch = None
    return keys.root_key(
        server_args.model_path,
        getattr(server_args, "component_quantizations", None),
        getattr(server_args, "transformer_quantization", None),
        epoch,
    )


def _collapsed_ids(stage, prompt, num_images):
    """Token ids of the collapsed encoder sequence (one row per image slot), as ``_encode_prompt`` builds it."""
    prefix = " ".join(
        f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(num_images)
    )
    text = (
        q21_stages.SYSTEM_TEMPLATE
        + f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    )
    ids = stage.processor.tokenizer(text)["input_ids"]
    return list(ids[stage.drop_idx :])


def encoding_forward(self, batch, server_args):
    config = server_args.pipeline_config
    ac = config.vae_config.arch_config
    device = get_local_torch_device()
    mt = MTRequest.from_batch(batch)  # mt
    images = batch.condition_image
    images = [] if images is None else images if isinstance(images, list) else [images]
    resized, shapes, conditions = [], [], []
    area = batch.height * batch.width
    image_mode = "RGBA" if ac.in_channels == 4 else "RGB"
    for image in images:
        if not isinstance(image, Image.Image):
            image = load_image(image, convert_method=lambda image: image.convert(image_mode))
        width = max(32, round(math.sqrt(area * image.width / image.height) / 32) * 32)
        height = max(32, round(math.sqrt(area * image.height / image.width) / 32) * 32)
        resized.append(image.convert(image_mode).resize((width, height), Image.Resampling.LANCZOS))
        shapes.append((1, height // 16, width // 16))

    # mt: condition latents from a stored output, then the VAE cache, then the VAE
    image_keys: list[bytes] = []
    conditions = [None] * len(resized)
    to_encode = []
    for index, image in enumerate(resized):
        latent_id = mt.ref_latents.get(index)
        stored = state.OUTPUT_LATENTS.get(latent_id) if latent_id else None
        if stored is not None and stored.shape[1] == shapes[index][1] * shapes[index][2]:
            conditions[index] = stored.to(device=device, dtype=torch.bfloat16)
            image_keys.append(keys.latent_key(latent_id))
            continue
        if latent_id:
            logger.warning("mtedit: latent %s missing or wrong shape; VAE-encoding reference %d", latent_id, index)
        content = keys.image_content_key(image)
        image_keys.append(content)
        cached = state.VAE_LATENTS.get(content)
        if cached is not None:
            conditions[index] = cached
        else:
            to_encode.append((index, image, content))

    if to_encode:
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            vae.use_tiling = config.vae_tiling
            for index, image, content in to_encode:
                pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(
                    image.height, image.width, ac.in_channels
                )
                # preserve the reference's batch stride for identical cuDNN convolution rounding
                pixels = pixels[None].permute(0, 3, 1, 2).unsqueeze(2).float() / 255.0
                pixels = (2 * pixels - 1).to(device=device, dtype=torch.bfloat16)
                latent = vae.encode(pixels).mode()
                mean = latent.new_tensor(ac.latents_mean).view(1, ac.z_dim, 1, 1, 1)
                std = latent.new_tensor(ac.latents_std).view(1, ac.z_dim, 1, 1, 1)
                conditions[index] = ((latent - mean) / std).flatten(2).transpose(1, 2)
                state.VAE_LATENTS.put(content, conditions[index])

    shapes.append((1, batch.height // 16, batch.width // 16))
    prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
    negatives = (
        batch.negative_prompt
        if isinstance(batch.negative_prompt, list)
        else [batch.negative_prompt] * len(prompts)
    )
    sample_count = len(prompts) * batch.num_outputs_per_prompt
    condition_latents = (
        torch.cat(conditions, dim=1).expand(sample_count, -1, -1) if conditions else None
    )

    use_store = state.PREFIX_CACHE and mt.prefix_cache and _store_supported(server_args)  # mt
    root = _root_key(server_args) if use_store else None
    image_shapes = [(h, w) for _, h, w in shapes[:-1]]

    for negative in [False, True] if batch.do_classifier_free_guidance else [False]:
        embeds, masks, layouts, chains = [], [], [], []
        for prompt in negatives if negative else prompts:
            with set_forward_context(current_timestep=None, attn_metadata=None, forward_batch=batch):
                hidden, slots = self.encode_prompt(prompt, resized, device)
            layout = build_layout(slots.tolist(), shapes, config.dit_config.axes_dims_rope, device)
            chain = None
            if use_store:  # mt
                chain = _chain_for(self, root, prompt, layout, hidden, image_keys, image_shapes)
            for _ in range(batch.num_outputs_per_prompt):
                embeds.append(hidden)
                layouts.append(layout)
                chains.append(chain)
        max_length = max(x.shape[0] for x in embeds)
        for x in embeds:
            masks.append(torch.arange(max_length, device=device) < x.shape[0])
        packed = torch.stack(
            [torch.nn.functional.pad(x, (0, 0, 0, max_length - x.shape[0])) for x in embeds]
        )
        mask = torch.stack(masks)
        if negative:
            batch.negative_prompt_embeds = [packed]
            batch.negative_prompt_embeds_mask = [mask]
            batch.negative_prompt_seq_lens = [mask.sum(1).tolist()]
        else:
            batch.prompt_embeds = [packed]
            batch.prompt_embeds_mask = [mask]
            batch.prompt_seq_lens = [mask.sum(1).tolist()]
        caches = [[{} for _ in range(config.dit_config.num_layers)] for _ in range(sample_count)]
        if use_store:  # mt
            caches = MTCaches(caches, chains, [l["prefix_rope"].shape[0] for l in layouts])
        batch.extra["qwen21_negative" if negative else "qwen21_positive"] = dict(
            layouts=layouts,
            condition_latents=condition_latents,
            prefix_caches=caches,
        )

    if mt.mask is not None:  # mt
        _prepare_mask(batch, mt, conditions, shapes)

    sched = self.scheduler.config
    batch.extra["qwen21_mu"] = calculate_linear_shift(
        (batch.height // 16) * (batch.width // 16),
        base_seq_len=sched.get("base_image_seq_len", 256),
        max_seq_len=sched.get("max_image_seq_len", 4096),
        base_shift=sched.get("base_shift", 0.5),
        max_shift=sched.get("max_shift", 1.15),
    )
    return batch


def _store_supported(server_args) -> bool:
    # the store keeps whole-prefix K/V per GPU; TP/SP shard heads or exchange K/V
    return (
        getattr(server_args, "tp_size", 1) in (None, 1)
        and getattr(server_args, "sp_degree", 1) in (None, 1)
        and getattr(server_args, "num_gpus", 1) in (None, 1)
    )


def _chain_for(stage, root, prompt, layout, hidden, image_keys, image_shapes):
    ids = _collapsed_ids(stage, prompt, len(image_keys))
    if len(ids) != hidden.shape[0] or len(ids) != layout["encoder_seq_len"]:
        logger.warning(
            "mtedit: token ids (%d) do not match encoder rows (%d); prefix store disabled for this prompt",
            len(ids),
            hidden.shape[0],
        )
        return None
    return keys.segment_chain(
        root,
        layout["segments"],
        layout["text_indices"].tolist(),
        ids,
        image_keys,
        image_shapes,
    )


def _prepare_mask(batch, mt, conditions, shapes):
    spec = mt.mask
    target = shapes[-1]
    if not (0 <= spec.canvas_ref < len(conditions)):
        raise ValueError(f"mask canvas_ref {spec.canvas_ref} is not a reference index")
    if shapes[spec.canvas_ref] != target:
        raise ValueError(
            "mask canvas reference must have the target size "
            f"({shapes[spec.canvas_ref][1:]} latent vs target {target[1:]})"
        )
    editable = rasterize(spec, batch.height, batch.width)
    batch.extra[MASK_EXTRA_KEY] = MaskState(
        x0=conditions[spec.canvas_ref][:1],
        editable=editable.view(1, -1, 1).to(conditions[spec.canvas_ref].device),
    )
