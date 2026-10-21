"""Fibo 2.0 Edit on SGLang-Diffusion: pipeline config, stages and registration.

Latents live as FLUX.2 packs them (2x2 patches, 128 channels, BatchNorm-normalized), which is also the token layout
Fibo's DiT embeds, so SGLang's FLUX.2 decode path is reused unchanged.
Per-request options in ``diffusers_kwargs={"fibo": {...}}``:
  mask_boxes       [[x0, y0, x1, y1], ...] in source pixels, greyed out (Fibo's masked edit)
  store_output_as  keep the output tokens under this id
  ref_latent       use stored tokens as the image to edit instead of VAE-encoding it (ignored with mask_boxes)
  ref_latent_crop  [x0, y0, x1, y1] in tokens, the crop of ``ref_latent`` a crop edit reads
  compose          {"base", "box", "as"}: paste a crop edit's tokens into ``base`` and store the result
  progressive      the first N steps run at half width and height
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch
from PIL import Image, ImageDraw

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLConfig
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.flux import Flux2PipelineConfig, _patchify_latents
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import ComponentUse
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import calculate_linear_shift
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import InputValidationStage
from sglang.multimodal_gen.runtime.utils.vision import load_image

from multi_turn.fibo.dit import FiboDitConfig

logger = logging.getLogger(__name__)
MODEL_PATHS = ["briaai/Fibo-2.0-edit-turbo"]
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
# Qwen3-VL layers behind the DiT's six text inputs: the Perceiver's, then one per injection block
OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TEXT_ENCODER_IMAGE_SIZE = 768
DEFAULT_AREA = 1024 * 1024


def parse_prompt(prompt: str):
    """``(caption, instruction)`` as Bria's pipeline reads a prompt: a VGL JSON object (``bria_vgl.vgl_json``) or plain
    text. Empty values and scores are dropped, as training did."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    def empty(value):
        if isinstance(value, float):
            return math.isnan(value)
        return value is None or (isinstance(value, (str, dict, list)) and not value)

    def clean(value):
        if isinstance(value, dict):
            value = {k: clean(v) for k, v in value.items()}
            return {k: v for k, v in value.items() if not empty(v)}
        if isinstance(value, list):
            return [v for v in map(clean, value) if not empty(v)]
        return value

    keys = set(caption)
    scores = ("aesthetic_score", "preference_score")
    caption = {
        key: {k: v for k, v in value.items() if k not in scores} if isinstance(value, dict) else value
        for key, value in clean(caption).items()
        if key not in scores
    }
    caption = {key: value for key, value in caption.items() if value != {}}
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "",
                   "text_render": caption.get("text_render") or []}
    return caption or None, instruction


def text_entries(caption) -> list[str]:
    entries = (caption.get("text_render") or []) if isinstance(caption, dict) else []
    return [e["text"] for e in entries if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"]]


def edit_prompt(prompt: str, num_images: int, text_images: bool = False) -> str:
    """The edit prompt Fibo 2.0 was trained with: a vision marker per image, the result's caption when the prompt has
    one, then the instruction; with text images, each text entry gets the marker of its image (as Bria's pipeline)."""
    caption, instruction = parse_prompt(prompt)
    if instruction is None:
        caption, instruction = None, prompt
    if text_images and isinstance(caption, dict):
        for entry in caption.get("text_render") or []:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
                entry["image_tokens"] = VISION_MARKER
    edit = {"image": VISION_MARKER} if num_images == 1 else {
        f"image_{k}": VISION_MARKER for k in range(1, num_images + 1)}
    if caption:
        edit["structured_caption"] = caption
    edit["edit_instruction"] = instruction
    return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)


@functools.lru_cache(maxsize=1)
def fallback_fonts():
    from huggingface_hub import snapshot_download

    from multi_turn.fibo.text_images import load_font

    root = snapshot_download(MODEL_PATHS[0], allow_patterns=["fonts/*"])
    return [load_font(os.path.join(root, "fonts", f)) for f in ("Aileron-Regular.otf", "DejaVuSans.ttf")]


def grey_out(image: Image.Image, boxes) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle([round(v) for v in box], fill=255)
    return Image.composite(Image.new("RGB", image.size, (128, 128, 128)), image.convert("RGB"), mask)


def resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = max(height / image.height, width / image.width)
    rw, rh = math.ceil(image.width * scale), math.ceil(image.height * scale)
    image = image.resize((rw, rh), resample=Image.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    return image.crop((left, top, left + width, top + height))


def options(batch) -> dict:
    kwargs = (getattr(batch, "extra", None) or {}).get("diffusers_kwargs") or {}
    return kwargs.get("fibo") or {}


@dataclass
class FiboSamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    num_frames: int = 1
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    negative_prompt: str | None = None

    @classmethod
    def default_image_output_format(cls) -> str:
        return "png"


@dataclass
class FiboPipelineConfig(Flux2PipelineConfig):
    native_only_components: tuple[str, ...] = ("transformer", "text_encoder", "vae")
    should_use_guidance: bool = False
    enable_autocast: bool = False
    generator_device: str = "cpu"
    dit_config: FiboDitConfig = field(default_factory=FiboDitConfig)
    vae_config: Flux2VAEConfig = field(default_factory=Flux2VAEConfig)
    text_encoder_configs: tuple = field(default_factory=lambda: (Qwen3VLConfig(),))
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    postprocess_text_funcs: tuple = (None,)
    preprocess_text_funcs: tuple = (None,)

    def supports_dynamic_batching(self):
        return False

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        return [float(s) for s in sigmas]

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        return (batch_size, self.dit_config.arch_config.num_channels_latents, batch.height // 16, batch.width // 16)

    def maybe_pack_latents(self, latents, batch_size, batch):
        return latents.flatten(2).transpose(1, 2)

    def maybe_prepare_latent_ids(self, latents):
        return None

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return batch.extra["fibo_cond"]

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return batch.extra["fibo_cond"]

    def slice_noise_pred(self, noise, latents):
        return noise

    def post_denoising_loop(self, latents, batch):
        batch.extra.pop("fibo_cond", None)
        h, w = batch.height // 16, batch.width // 16
        opts = options(batch)
        if opts.get("store_output_as") or opts.get("compose"):
            from multi_turn import state
            from multi_turn.pipeline.hooks import _compose

            if opts.get("store_output_as"):
                state.OUTPUT_LATENTS.put(opts["store_output_as"], latents[:1].detach().to(torch.bfloat16).clone())
                state.OUTPUT_GRID[opts["store_output_as"]] = (h, w)
            if opts.get("compose"):
                _compose(opts["compose"], latents[:1], (h, w))
        return latents.transpose(1, 2).reshape(latents.shape[0], -1, h, w)

    def preprocess_condition_image(self, image, *args, **kwargs):
        return image


class FiboInputValidationStage(InputValidationStage):
    def preprocess_condition_image(self, batch, server_args, condition_image_width, condition_image_height):
        return None

    def forward(self, batch, server_args):
        if batch.prompt is None:
            raise ValueError("Fibo 2.0 needs an edit instruction")
        batch = super().forward(batch, server_args)
        if batch.height % 16 or batch.width % 16:
            raise ValueError("Fibo 2.0 height and width must be divisible by 16")
        batch.sigmas = server_args.pipeline_config.prepare_sigmas(batch.sigmas, batch.num_inference_steps)
        batch.num_inference_steps = len(batch.sigmas)
        return batch


class FiboEncodingStage(PipelineStage):
    def __init__(self, text_encoder, processor, vae, scheduler, transformer):
        super().__init__()
        self.text_encoder, self.processor, self.vae, self.scheduler = text_encoder, processor, vae, scheduler
        self.transformer = transformer

    def component_uses(self, server_args, stage_name=None):
        name = self._component_stage_name(stage_name)
        return [ComponentUse(name, "text_encoder"), ComponentUse(name, "vae", target_dtype=torch.bfloat16)]

    def encode_prompt(self, prompt, images, device):
        from multi_turn.fibo.text_images import draw_text_image

        # text_render entries are shown to Qwen3-VL as drawn images after the images to edit, at every edit step
        texts = text_entries(parse_prompt(prompt)[0])
        text_images = [draw_text_image(t, fallback_fonts()) for t in texts]
        body = edit_prompt(prompt, len(images), bool(text_images)) if images else prompt
        inputs = self.processor(text=[PROMPT_TEMPLATE.format(body)], images=(images + text_images) or None,
                                max_length=4096, truncation=True, return_tensors="pt").to(device)
        with self.use_declared_component(component_name="text_encoder", module=self.text_encoder) as encoder:
            hidden = encoder(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        normalized = {}
        for layer in sorted({layer for layers in OUT_LAYERS for layer in layers}):
            states = hidden[layer].float()
            normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        self.finish_unused_declared_component(component_name="text_encoder", module=self.text_encoder)
        return torch.stack([torch.cat([normalized[layer] for layer in layers], dim=-1) for layers in OUT_LAYERS], 1)

    def encode_images(self, crops, config, device):
        tokens, hw = [], []
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            for crop in crops:
                pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
                latents = vae.encode(pixels.contiguous().to(device, torch.bfloat16)).mode().float()
                latents = config.normalize_vae_encode(_patchify_latents(latents), vae)
                hw.append(tuple(latents.shape[2:]))
                tokens.append(latents.flatten(2).transpose(1, 2))
        return torch.cat(tokens, dim=1), hw

    def forward(self, batch, server_args):
        from multi_turn.fibo.fp4 import maybe_convert

        # before the first VAE encode, so the BF16 DiT is gone by the time a 2K encode needs the room
        maybe_convert(self.transformer)
        config = server_args.pipeline_config
        device = get_local_torch_device()
        images = batch.condition_image
        images = [] if images is None else images if isinstance(images, list) else [images]
        images = [im if isinstance(im, Image.Image) else load_image(im) for im in images]
        boxes = options(batch).get("mask_boxes")
        if boxes and images:
            images[0] = grey_out(images[0], boxes)
        opts = options(batch)
        stored = None
        if opts.get("ref_latent") and not boxes:
            from multi_turn import state

            stored = state.stored_latents(opts["ref_latent"], opts.get("ref_latent_crop"))
        crops = [resize_and_crop(im, batch.width, batch.height) for im in images[:1]]
        crops += [resize_and_crop(im, *_trained_size(im)) for im in images[1:]]
        vlm_images = []
        for crop in crops:
            copy = crop.copy()
            copy.thumbnail((TEXT_ENCODER_IMAGE_SIZE, TEXT_ENCODER_IMAGE_SIZE))
            vlm_images.append(copy)

        prompt = batch.prompt[0] if isinstance(batch.prompt, list) else batch.prompt
        with set_forward_context(current_timestep=None, attn_metadata=None, forward_batch=batch):
            embeds = self.encode_prompt(prompt, vlm_images, device)
        batch.prompt_embeds = [embeds]
        batch.prompt_embeds_mask = None
        cond = {"latent_hw": [batch.height // 16, batch.width // 16]}
        if stored is not None and len(crops) == 1:
            cond["context_tokens"] = stored.to(device)
            cond["context_hw"] = [(batch.height // 16, batch.width // 16)]
        elif crops:
            cond["context_tokens"], cond["context_hw"] = self.encode_images(crops, config, device)
        steps = int(opts.get("progressive") or 0)
        if steps and not boxes and batch.height % 32 == 0 and batch.width % 32 == 0:
            batch.extra["fibo_progressive"] = steps
        batch.extra["fibo_cond"] = cond
        sched = self.scheduler.config
        batch.extra["fibo_mu"] = calculate_linear_shift(
            (batch.height // 16) * (batch.width // 16),
            base_seq_len=sched.get("base_image_seq_len", 256),
            max_seq_len=sched.get("max_image_seq_len", 4096),
            base_shift=sched.get("base_shift", 0.5),
            max_shift=sched.get("max_shift", 1.15),
        )
        return batch


def _trained_size(image, area=DEFAULT_AREA, multiple=16):
    return trained_size(image.width, image.height, area, multiple)


def trained_size(width, height, area=DEFAULT_AREA, multiple=16):
    """The (width, height) Fibo 2.0 was trained at for this aspect ratio, about ``area`` (Bria's get_image_size)."""
    aspect, min_side = width / height, 4 * multiple
    sizes = [(w, h) for w in range(min_side, 2 * area // min_side, multiple)
             for h in range(min_side, 2 * area // w, multiple) if 0.9 * area <= w * h <= 1.1 * area]
    return min(sizes, key=lambda s: (round(1 - min(s[0] / s[1] / aspect, aspect / (s[0] / s[1])), 9),
                                     abs(s[0] * s[1] - area), -s[0]))


def prepare_fibo_mu(batch, server_args):
    return "mu", batch.extra["fibo_mu"]


class FiboDenoisingStage(DenoisingStage):
    def _predict_noise(self, current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs):
        from multi_turn.fibo.fp4 import maybe_convert

        maybe_convert(current_model)
        return super()._predict_noise(current_model, latent_model_input, timestep, target_dtype, guidance, **kwargs)

    def _run_denoising_step(self, ctx, step, batch, server_args):
        steps = batch.extra.get("fibo_progressive", 0)
        if step.step_index == 0 and os.environ.get("MULTI_TURN_DEBUG") == "1":
            logger.info("multi_turn: fibo %dx%d sigmas %s", batch.width, batch.height,
                        [round(float(x), 4) for x in ctx.scheduler.sigmas])
        if step.step_index >= steps:
            return super()._run_denoising_step(ctx, step, batch, server_args)
        # as Qwen's progressive path: half-size target first, then upsample x0 and re-noise with the same noise
        cond = batch.extra["fibo_cond"]
        H, W = batch.height // 16, batch.width // 16
        if step.step_index == 0:
            self._eps = ctx.latents.detach().clone()
            b, _, c = ctx.latents.shape
            ctx.latents = ctx.latents.view(b, H, W, c)[:, ::2, ::2].reshape(b, (H // 2) * (W // 2), c).contiguous()
            cond["latent_hw"][:] = [H // 2, W // 2]  # in place: the denoising kwargs copy the dict, not the list
        before = ctx.latents
        result = super()._run_denoising_step(ctx, step, batch, server_args)
        if step.step_index == steps - 1:
            sigmas = ctx.scheduler.sigmas
            s0, s1 = sigmas[step.step_index], sigmas[step.step_index + 1]
            x0 = ctx.latents - s1 * (ctx.latents - before) / (s1 - s0)
            b, _, c = x0.shape
            x0 = x0.view(b, H // 2, W // 2, c).permute(0, 3, 1, 2)
            x0 = torch.nn.functional.interpolate(x0.float(), size=(H, W), mode="bilinear", align_corners=False)
            x0 = x0.permute(0, 2, 3, 1).reshape(b, H * W, c).to(ctx.latents.dtype)
            ctx.latents = (1 - s1) * x0 + s1 * self._eps.to(x0)
            cond["latent_hw"][:] = [H, W]
        return result


class FiboPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "FiboEditPipeline"
    pipeline_config_cls = FiboPipelineConfig
    sampling_params_cls = FiboSamplingParams
    _required_config_modules = ["processor", "text_encoder", "transformer", "vae", "scheduler"]

    def create_pipeline_stages(self, server_args):
        self.add_stage(FiboInputValidationStage())
        self.add_stage(FiboEncodingStage(self.get_module("text_encoder"), self.get_module("processor"),
                                         self.get_module("vae"), self.get_module("scheduler"),
                                         self.get_module("transformer")))
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(prepare_extra_kwargs=[prepare_fibo_mu])
        self.add_stage(FiboDenoisingStage(transformer=self.get_module("transformer"),
                                          scheduler=self.get_module("scheduler")))
        self.add_standard_decoding_stage()


def register() -> None:
    from sglang.multimodal_gen.registry import register_pipeline
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    ModelRegistry.register_model("BriaFibo2Transformer2DModel", "multi_turn.fibo.dit:FiboTransformer2DModel")
    register_pipeline(
        FiboPipeline,
        sampling_param_cls=FiboSamplingParams,
        pipeline_config_cls=FiboPipelineConfig,
        hf_model_paths=MODEL_PATHS,
        model_detectors=[lambda path: "fibo-2.0" in path.lower()],
    )
