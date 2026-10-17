"""Client-side multi-turn edit sessions over SGLang's ``DiffGenerator``.

A session is a version tree. Each edit builds a normal Qwen-Image 2.1 edit
request whose references follow a fixed slot layout, so the prefix K/V of
unchanged references (and of a whole re-rolled request) is reused by the
worker-side store:

  chain   refs = [current]               one image, most in-distribution; default
  anchor  refs = [original, current]     keeps identity from the original; the
                                         original's K/V is cached once per session

Region edits pass boxes; the worker keeps everything outside them on the source
latents and the session pastes the source pixels back outside a feathered mask.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter

Box = tuple[float, float, float, float]

# Boxed edits: the worker outlines the boxes in red on the canvas the model sees
# (Qwen-Image 2.1's mark-guided local editing) and the prompt names the mark.
BOX_TEMPLATE = (
    "Edit only the area inside the red box: {instruction} "
    "Remove the red box outline completely and keep everything outside the box unchanged."
)

# Turbo's distilled schedule and subsets of it (5 and 4 steps measured close to 8 on the photo suite)
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
SPEED_PRESETS = {
    "quality": None,  # the model's own schedule
    "fast": [1.0, 0.926626, 0.845148, 0.704534, 0.414568],
    "draft": [1.0, 0.89508, 0.704534, 0.414568],
}

# Multi-region edits: each box gets its own outline color and instruction.
MARK_PALETTE = [
    ("red", (255, 0, 0)), ("blue", (0, 90, 255)), ("green", (0, 200, 60)), ("yellow", (255, 220, 0)),
    ("magenta", (255, 0, 255)), ("cyan", (0, 220, 255)), ("orange", (255, 140, 0)), ("purple", (140, 60, 220)),
]
REGION_TEMPLATE = "{regions}{instruction} Remove all colored box outlines completely and keep everything outside the boxes unchanged."

ANCHOR_TEMPLATE = (
    "<image1> is the original image and <image2> is its current edited version. "
    "Edit <image2>: {instruction} Keep everything else in <image2> unchanged."
)


@dataclass
class Version:
    id: str
    parent: str | None
    image_path: str
    instruction: str | None = None
    boxes: list[Box] | None = None
    regions: list | None = None  # [(box, instruction)] for multi-region edits
    feather_px: float = 16.0
    grow_px: float = 0.0
    seed: int | None = None
    latent_id: str | None = None  # worker-side final latents, for passthrough
    seconds: float = 0.0
    timings: dict = field(default_factory=dict)  # generate / post / save
    prompt: str | None = None  # the exact prompt sent to the model
    resolved: dict | None = None  # context resolver output: instruction, mode, target
    change_box: Box | None = None  # where this edit actually changed the image (scene memory)
    request: str | None = None  # what the user typed
    base: str | None = None  # the version the model actually edited (the anchor for text edits)
    standalone: bool = False  # a typed edit that is not a follow-up: later follow-ups do not redo it
    children: list[str] = field(default_factory=list)


class Engine:
    """One ``DiffGenerator`` (and its GPU worker) shared by any number of sessions."""

    def __init__(self, model_path: str = "Qwen/Qwen-Image-2.1-Turbo", out_dir: str = "mt_out",
                 attention: str | None = None, **server_kwargs):
        from sglang.multimodal_gen import DiffGenerator

        server_kwargs.setdefault("performance_mode", "speed")
        self.model_path = model_path
        self.attention = attention  # default DiT attention override for every request
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.generator = DiffGenerator.from_pretrained(model_path=model_path, **server_kwargs)
        self.lock = threading.Lock()  # one generation at a time (edits and background warm-ups)

    def run(self, *, prompt, images, width, height, seed, mt, name, steps=None, guidance_scale=None, save=False,
            attention=None):
        """One generation; returns (PIL image, result). ``save=False`` keeps the output in memory.

        ``attention`` switches only the DiT's attention for this request (e.g. ``"sage_attn"``);
        the text encoder keeps its backend.
        """
        kwargs = dict(
            prompt=prompt,
            image_path=list(images) or None,
            width=width,
            height=height,
            seed=seed,
            save_output=save,
            output_path=self.out_dir,
            output_file_name=name,
            diffusers_kwargs={"mt": mt},
        )
        if steps is not None:
            kwargs["num_inference_steps"] = steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        attention = attention or self.attention
        if attention:
            kwargs["attention_backend_override"] = attention
        with self.lock:
            result = self.generator.generate(sampling_params_kwargs=kwargs)
        result = result[0] if isinstance(result, list) else result
        if result is None:
            raise RuntimeError("generation failed")
        if save:
            return Image.open(result.output_file_path), result
        return _to_pil(result), result

    def text_to_image(self, prompt, *, width=1024, height=1024, seed=None, steps=None):
        """Generate a base image; returns (image, latent_id) so a session can start from its latents."""
        seed = int(time.time_ns() % (2**31)) if seed is None else seed
        latent_id = f"t2i/{uuid.uuid4().hex[:10]}"
        image, _ = self.run(prompt=prompt, images=[], width=width, height=height, seed=seed,
                            mt={"store_output_as": latent_id}, name=f"t2i_{latent_id[4:]}", steps=steps)
        return image, latent_id

    def shutdown(self):
        self.generator.shutdown()


def _to_pil(result) -> Image.Image:
    """The in-memory output of an unsaved generation as a PIL image."""
    import numpy as np

    frame = result.frames[0] if result.frames else result.samples
    if hasattr(frame, "detach"):  # a tensor: [C, H, W] or [C, 1, H, W] in [0, 1]
        frame = frame.detach().float().cpu()
        while frame.dim() > 3:
            frame = frame[:, 0] if frame.shape[1] == 1 else frame[0]
        frame = (frame.clamp(0, 1) * 255).round().to(dtype=__import__("torch").uint8).permute(1, 2, 0).numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(frame)


_REFERRING = re.compile(
    r"\b(it|its|them|they|their|that|those|this|these|him|her|his|he|she|one|ones|again|instead|"
    r"more|less|bigger|smaller|larger|darker|lighter|brighter|redder|longer|shorter|wider|taller|too|"
    r"same|other|rather)\b", re.I)


def _may_refer(request: str) -> bool:
    """Cheap gate before asking the resolver whether this is a follow-up of the last edit."""
    return len(request.split()) <= 4 or bool(_REFERRING.search(request))


def _pad_box(box, pad, size):
    x0, y0, x1, y1 = box
    return (max(0, x0 - pad), max(0, y0 - pad), min(size[0], x1 + pad), min(size[1], y1 + pad))


def _round32(value: float) -> int:
    return max(32, int(round(value / 32)) * 32)


def change_mask(new: Image.Image, old: Image.Image, threshold: float = 14.0, dilate_px: int = 24,
                min_area_frac: float = 0.0005, box_only: bool = False):
    """Where ``new`` differs from ``old``: (feathered L mask, bounding box) or (None, None).

    Computed on a 1/8 grid (fast, ignores pixel noise), then dilated so the kept
    region includes the object's shadow and blend zone.
    """
    import numpy as np
    from PIL import ImageFilter

    w, h = new.size
    small = (max(1, w // 8), max(1, h // 8))
    a = np.asarray(new.convert("RGB").resize(small, Image.Resampling.BILINEAR), np.float32)
    b = np.asarray(old.convert("RGB").resize(small, Image.Resampling.BILINEAR), np.float32)
    diff = np.abs(a - b).mean(-1)
    hit = diff > max(threshold, 0.35 * float(diff.max()))  # strong changes only, not broad relighting
    if hit.mean() < min_area_frac:
        return None, None
    try:  # keep the connected blob around the strongest change (the object, not stray drift)
        from scipy import ndimage

        labels, n = ndimage.label(ndimage.binary_closing(hit, iterations=2))
        if n > 1:
            peak = np.unravel_index(np.argmax(np.where(hit, diff, 0)), diff.shape)
            hit = labels == labels[peak]
    except ImportError:
        pass
    ys, xs = np.nonzero(hit)
    box = (int(xs.min()) * 8, int(ys.min()) * 8, (int(xs.max()) + 1) * 8, (int(ys.max()) + 1) * 8)
    if box_only:  # the full-resolution mask (a slow max-filter at 4 MP) is only needed for compositing
        return None, box
    mask = Image.fromarray((hit * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    mask = mask.filter(ImageFilter.MaxFilter(2 * (dilate_px // 2) + 1)).filter(ImageFilter.GaussianBlur(dilate_px / 3))
    return mask, box


def blend_box_edit(out: Image.Image, source: Image.Image, boxes, feather_px: float, grow_px: float,
                   band_px: int = 12, keep_lo: float = 6.0, keep_hi: float = 22.0) -> Image.Image:
    """Paste a boxed edit back so it matches its surroundings.

    1. tone-match: per channel, map the generated box's colors so that a thin band just
       inside the box edge (where nothing should change) matches the original there;
    2. keep only real changes: inside the box, pixels the model merely re-rendered
       (small difference to the original) revert to the original; only the new or
       changed content is pasted, with a soft edge.
    """
    import numpy as np
    from PIL import ImageFilter

    # do all the work inside the boxes' padded union: outside it the result is the source
    pad = int(feather_px + grow_px + 16)
    W, H = out.size
    ux0 = max(0, int(min(b[0] for b in boxes)) - pad); uy0 = max(0, int(min(b[1] for b in boxes)) - pad)
    ux1 = min(W, int(max(b[2] for b in boxes)) + pad); uy1 = min(H, int(max(b[3] for b in boxes)) + pad)
    if (ux0, uy0, ux1, uy1) != (0, 0, W, H):
        local = [(b[0] - ux0, b[1] - uy0, b[2] - ux0, b[3] - uy0) for b in boxes]
        patch = blend_box_edit(out.crop((ux0, uy0, ux1, uy1)), source.crop((ux0, uy0, ux1, uy1)), local,
                               feather_px, grow_px, band_px, keep_lo, keep_hi)
        full = source.copy()
        full.paste(patch, (ux0, uy0))
        return full

    o = np.asarray(out.convert("RGB"), np.float32)
    src = np.asarray(source.convert("RGB"), np.float32)
    h, w = o.shape[:2]
    result = o.copy()
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 - x0 < 4 * band_px or y1 - y0 < 4 * band_px:
            continue
        region = (slice(y0, y1), slice(x0, x1))
        ring = np.ones((y1 - y0, x1 - x0), bool)
        ring[band_px:-band_px, band_px:-band_px] = False
        go, gs = o[region][ring], src[region][ring]
        mo, ms = go.mean(0), gs.mean(0)
        so, ss = go.std(0) + 1e-3, gs.std(0) + 1e-3
        scale = np.clip(ss / so, 0.8, 1.25)  # correct tone, never invent contrast
        result[region] = np.clip((o[region] - mo) * scale + ms, 0, 255)
    # where the result barely differs from the original, keep the original
    diff = np.abs(result - src).mean(-1)
    diff = np.asarray(Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32)
    changed = np.clip((diff - keep_lo) / (keep_hi - keep_lo), 0, 1)
    changed = np.asarray(Image.fromarray((changed * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
                         .filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
    box = np.asarray(pixel_mask(out.size, boxes, feather_px, grow_px), np.float32) / 255
    m = (box * changed)[..., None]
    blended = (result * m + src * (1 - m)).astype(np.uint8)
    final = Image.fromarray(blended)
    return final.convert(source.mode) if source.mode != "RGB" else final


def pixel_mask(size: tuple[int, int], boxes: list[Box], feather_px: float, grow_px: float) -> Image.Image:
    """Feathered editable mask in pixels (255 = take the new image)."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    pad = grow_px  # the blur is centered on the box edge: it feathers inward and outward
    for x0, y0, x1, y1 in boxes:
        draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather_px / 2)) if feather_px > 0 else mask


class EditSession:
    def __init__(
        self,
        engine: Engine,
        base_image: str | Image.Image,
        *,
        layout: str = "chain",
        size: tuple[int, int] | None = None,
        template: str | None = None,
        passthrough: bool = True,
        paste_back: bool = True,
        steps: int | None = None,
        guidance_scale: float | None = None,
        session_id: str | None = None,
        cache: bool = True,
        mark_boxes: bool = True,
        box_template: str | None = None,
        prewarm: bool = False,
        root_latent_id: str | None = None,
        crop: bool = True,
        crop_margin_px: int = 96,
        crop_min_px: int = 512,
        crop_max_frac: float = 0.55,
        sigmas: list[float] | None = None,
        megapixels: float = 1.0,
        ref_scale: float = 1.0,
    ):
        if layout not in ("chain", "anchor"):
            raise ValueError("layout must be 'chain' or 'anchor'")
        self.engine = engine
        self.layout = layout
        self.template = template or (ANCHOR_TEMPLATE if layout == "anchor" else "{instruction}")
        self.passthrough = passthrough
        self.paste_back = paste_back
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.cache = cache  # False: no prefix/encoder reuse (reference runs)
        self.mark_boxes = mark_boxes
        self.box_template = box_template or BOX_TEMPLATE
        self.auto_prewarm = prewarm
        # boxed edits generate only a crop around the boxes (+ context margin) when it is small enough
        self.sigmas = sigmas  # explicit schedule, e.g. a Turbo subset for drafts
        self.ref_scale = ref_scale  # unboxed edits: reference area / target area
        self.progressive = 0  # unboxed edits: first N steps at half resolution
        self.attention: str | None = None  # DiT-only attention override, e.g. "sage_attn"
        self.resolver = None  # mtedit.context.Resolver: makes follow-ups self-contained
        self.modify_start_sigma = 0.85  # SDEdit start for boxed modify edits (opt-in)
        self.sdedit_modify = False  # Turbo's distilled schedule leaves grain when started mid-way
        # unboxed text edits re-edit the anchor (root or latest boxed result) with all requests
        # since it merged into one instruction: no compounding artifacts across turns
        self.cumulative = True
        self.crop, self.crop_margin_px, self.crop_min_px, self.crop_max_frac = crop, crop_margin_px, crop_min_px, crop_max_frac  # warm the next turn's prefix in the background after each edit
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mtedit-{session_id or 'io'}")
        self._saves: dict[str, Future] = {}
        self._warm: Future | None = None
        self.id = session_id or uuid.uuid4().hex[:8]
        self.dir = os.path.join(engine.out_dir, f"session_{self.id}")
        os.makedirs(self.dir, exist_ok=True)

        image = base_image if isinstance(base_image, Image.Image) else Image.open(base_image)
        if size is None:  # keep the aspect at the requested megapixels (the model card edits at ~4 MP), 32-multiples
            scale = (megapixels * 1024 * 1024 / (image.width * image.height)) ** 0.5
            size = (_round32(image.width * scale), _round32(image.height * scale))
        self.size = size
        # every version, including the root, lives at the target size so slots keep their shapes
        root = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
        root = root.resize(size, Image.Resampling.LANCZOS)
        path = os.path.join(self.dir, "v000.png")
        root.save(path, compress_level=1)
        self._image_cache = {"v000": root}
        # a generated base keeps its latents for the first edit (only valid at its native size)
        native = root_latent_id if image.size == tuple(size) else None
        self.versions: dict[str, Version] = {
            "v000": Version(id="v000", parent=None, image_path=path, latent_id=native)
        }
        self.root = self.head = "v000"
        self._counter = 0

    # history ---------------------------------------------------------------

    @property
    def current(self) -> Version:
        return self.versions[self.head]

    def image(self, version: str | None = None) -> Image.Image:
        vid = version or self.head
        cached = self._image_cache.get(vid)
        return cached if cached is not None else Image.open(self._file(vid))

    def path(self, vid: str | None = None) -> str:
        """The version's image file, once its background save has finished (always use this, not
        ``Version.image_path``, to read a version's file)."""
        vid = vid or self.head
        pending = self._saves.pop(vid, None)
        if pending is not None:
            pending.result()
        return self.versions[vid].image_path

    _file = path

    def flush(self) -> None:
        """Wait for every pending save and warm-up."""
        for vid in list(self._saves):
            self.path(vid)
        if self._warm is not None:
            self._warm.result()

    def _refs(self, parent: "Version") -> list["Version"]:
        if self.layout == "anchor" and parent.id != self.root:
            return [self.versions[self.root], parent]
        return [parent]

    def prewarm(self, version: str | None = None, background: bool = True):
        """Prefill the next turn's prefix (references + template head) while the user is thinking.

        The next unboxed edit from ``version`` then computes only its prompt tail.
        Boxed edits outline the boxes on the canvas, so their canvas cannot be warmed.
        """
        parent = self.versions[version or self.head]
        refs = self._refs(parent)
        template = self.template if len(refs) == 2 else "{instruction}"
        mt = {"prefill_only": True, "prefix_cache": True}
        if self.passthrough:
            mt["ref_latents"] = {i: r.latent_id for i, r in enumerate(refs) if r.latent_id}
            # references without stored latents (an uploaded root) get them now, for crop edits and passthrough
            missing = {i: f"{self.id}/{r.id}" for i, r in enumerate(refs) if not r.latent_id}
            if missing:
                mt["store_ref_as"] = missing

        def run():
            self.engine.run(
                prompt=template.format(instruction=""), images=[self._file(r.id) for r in refs],
                width=self.size[0], height=self.size[1], seed=0, mt=mt, name=f"session_{self.id}_warm",
                steps=self.steps, guidance_scale=self.guidance_scale,
            )
            for i, name in mt.get("store_ref_as", {}).items():  # only once the worker holds them
                refs[i].latent_id = name

        if not background:
            return run()
        self._warm = self._io.submit(run)
        return self._warm

    def undo(self) -> Version:
        if self.current.parent is not None:
            self.head = self.current.parent
        return self.current

    def checkout(self, version: str) -> Version:
        if version not in self.versions:
            raise KeyError(version)
        self.head = version
        return self.current

    def lineage(self, version: str | None = None) -> list[Version]:
        out, vid = [], version or self.head
        while vid is not None:
            out.append(self.versions[vid])
            vid = self.versions[vid].parent
        return out[::-1]

    # editing -----------------------------------------------------------------

    def crop_rect(self, boxes: list[Box]) -> tuple[int, int, int, int] | None:
        """Pixel window for a crop edit: the boxes' union plus margin, on the 16 px latent grid with
        32 px-multiple sides; None when it would cover too much of the image to be worth it."""
        W, H = self.size
        ux0, uy0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        ux1, uy1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        margin = max(self.crop_margin_px, 0.35 * max(ux1 - ux0, uy1 - uy0))

        def span(lo, hi, limit):
            lo, hi = max(0.0, lo - margin), min(float(limit), hi + margin)
            need = min(max(self.crop_min_px, hi - lo), limit)
            grow = need - (hi - lo)
            lo, hi = lo - grow / 2, hi + grow / 2
            if lo < 0:
                lo, hi = 0, hi - lo
            if hi > limit:
                lo, hi = lo - (hi - limit), limit
            lo = max(0, int(lo // 16) * 16)
            size = min(-(-int(hi - lo) // 32) * 32, limit - limit % 32)
            lo = min(lo, limit - size) // 16 * 16
            return lo, lo + size

        x0, x1 = span(ux0, ux1, W)
        y0, y1 = span(uy0, uy1, H)
        if (x1 - x0) * (y1 - y0) > self.crop_max_frac * W * H:
            return None
        return x0, y0, x1, y1

    def edit(
        self,
        instruction: str = "",
        *,
        boxes: list[Box] | None = None,
        regions: list[tuple[Box, str]] | None = None,
        seed: int | None = None,
        feather_px: float = 16.0,
        grow_px: float = 0.0,
        from_version: str | None = None,
        raw_prompt: str | None = None,
        sigmas: list[float] | None = None,
        attention: str | None = None,
    ) -> Version:
        """Apply one edit to ``from_version`` (default: head) and make the result the head.

        ``raw_prompt`` (e.g. from the prompt enhancer) is sent as is instead of the templated
        instruction; boxes and regions still drive the marks and the mask.
        """
        parent = self.versions[from_version or self.head]
        base, request = parent, (instruction or "").strip()
        merged = None
        if (self.cumulative and self.resolver is not None and request and not boxes and not regions
                and not raw_prompt):
            # a follow-up of the LAST typed edit ("make them blue" after "give him red sneakers") redoes
            # only that edit, from its own base, with both merged; anything else edits the current image,
            # so earlier edits you liked are never redone
            last = parent
            if last.parent is not None and last.request and not last.boxes and not last.regions \
                    and _may_refer(request):
                try:
                    probe = self.resolver.resolve([(1, last.instruction or last.request)], request)
                    if probe.get("target") == 1:
                        merged = self.resolver.merge([last.request if last.base == last.parent else
                                                      (last.instruction or last.request), request])
                        base = self.versions[last.base or last.parent]
                except Exception as err:
                    merged = {"error": str(err)}
        refs = self._refs(base)
        canvas_ref = len(refs) - 1
        template = self.template if len(refs) == 2 else "{instruction}"
        instruction = (instruction or "").strip()
        resolved, sdedit = None, False
        if merged and "instruction" in merged:
            instruction, resolved = merged["instruction"], {"mode": "merged", "base": base.id, **merged}
        elif self.resolver is not None and instruction and not regions and not raw_prompt:
            lineage = [v for v in self.lineage(parent.id) if v.parent is not None]
            history = [(i + 1, v.instruction or "; ".join(t for _, t in (v.regions or []))) for i, v in enumerate(lineage)]
            try:
                resolved = self.resolver.resolve(history, instruction)
            except Exception as err:  # the resolver is an aid, never a blocker
                resolved = {"error": str(err)}
            if "instruction" in resolved:
                instruction = resolved["instruction"]
                # no silent box reuse: an edit without a drawn box has no box (and no box wording)
                sdedit = self.sdedit_modify and resolved.get("mode") == "modify" and bool(boxes)
        colors = None
        if regions:  # per-box instructions: one outline color per box, named in the prompt
            if len(regions) > len(MARK_PALETTE):
                raise ValueError(f"at most {len(MARK_PALETTE)} regions per edit")
            boxes = [tuple(map(float, box)) for box, _ in regions]
            parts = [f"Inside the {MARK_PALETTE[i][0]} box: {text.strip().rstrip('.')}." for i, (_, text) in enumerate(regions)]
            colors = [MARK_PALETTE[i][1] for i in range(len(regions))]
            task = REGION_TEMPLATE.format(regions=" ".join(parts) + " ", instruction=instruction).replace("  ", " ")
        elif boxes and self.mark_boxes:
            # never outline in a color the instruction names ("make it red" gets a blue box)
            free = [(n, c) for n, c in MARK_PALETTE if n not in instruction.lower()] or MARK_PALETTE
            mark_name, mark_rgb = free[0]
            colors = [mark_rgb] * len(boxes)
            task = self.box_template.replace("red box", f"{mark_name} box").format(instruction=instruction)
        else:
            task = instruction
        if raw_prompt:
            task = raw_prompt.strip()
        prompt = template.format(instruction=task)

        self._counter += 1
        vid = f"v{self._counter:03d}"
        latent_id = f"{self.id}/{vid}"
        sigmas = sigmas if sigmas is not None else self.sigmas
        if sdedit:  # keep geometry: start from the noised source, skip the high-noise steps
            sigmas = [s for s in (sigmas or TURBO_SIGMAS) if s <= self.modify_start_sigma] or None
        attention = attention or self.attention
        mt: dict = {"store_output_as": latent_id, "prefix_cache": self.cache}
        if self.ref_scale != 1.0 and not boxes:
            mt["ref_scale"] = self.ref_scale
        if self.progressive and not boxes:
            mt["progressive"] = self.progressive
        if sigmas:
            mt["sigmas"] = list(sigmas)
        if self.passthrough:
            mt["ref_latents"] = {i: r.latent_id for i, r in enumerate(refs) if r.latent_id}
        if boxes:
            mt["mask"] = {
                "boxes": [list(b) for b in boxes], "canvas_ref": canvas_ref, "feather_px": feather_px,
                "grow_px": grow_px, "annotate": self.mark_boxes or bool(regions),
                "init_from_source": sdedit,
            }
            if colors:
                mt["mask"]["colors"] = [list(c) for c in colors]
        seed = int(time.time_ns() % (2**31)) if seed is None else seed
        source = self._image_cache.get(base.id) or Image.open(self.path(base.id))
        window = self.crop_rect(boxes) if boxes and self.crop else None

        start = time.perf_counter()
        if window is None:
            out, result = self.engine.run(
                prompt=prompt, images=[self._file(r.id) for r in refs], width=self.size[0], height=self.size[1],
                seed=seed, mt=mt, name=f"session_{self.id}_{vid}_raw", steps=self.steps,
                guidance_scale=self.guidance_scale, attention=attention,
            )
            generated = time.perf_counter()
            out = out.convert(source.mode)
            if boxes and self.paste_back:
                out = blend_box_edit(out, source, boxes, feather_px, grow_px)
            # unboxed edits are kept whole: soft effects (glow, shadows, reflections) do not survive
            # a hard change mask; the change box is still measured below for scene memory
        else:
            out, result, latent_id = self._crop_edit(
                parent, source, window, task, boxes, colors, regions, seed, feather_px, grow_px, vid, latent_id,
                sigmas, attention, sdedit,
            )
            generated = time.perf_counter()
        posted = time.perf_counter()
        path = os.path.join(self.dir, f"{vid}.png")
        self._image_cache[vid] = out
        # the next turn reads the file; level 1 is ~10x faster than the default and it is off the critical path
        self._saves[vid] = self._io.submit(out.save, path, compress_level=1)
        seconds = time.perf_counter() - start
        timings = dict(generate=generated - start, post=posted - generated, save=time.perf_counter() - posted,
                       crop=list(window) if window else None)

        change_box = change_mask(out, source, box_only=True)[1] if self.paste_back else None
        version = Version(
            id=vid, parent=parent.id, image_path=path, instruction=instruction, regions=regions,
            boxes=boxes, feather_px=feather_px, grow_px=grow_px, seed=seed,
            latent_id=latent_id, seconds=seconds, timings=timings, prompt=prompt, resolved=resolved,
            change_box=change_box, request=request or None, base=base.id,
            standalone=bool(request) and not boxes and not regions and merged is None,
        )
        self.versions[vid] = version
        parent.children.append(vid)
        self.head = vid
        # prewarm only where it is cheap: at 2K it holds the GPU ~1-1.5 s and a quick next edit waits on it
        if self.auto_prewarm and self.size[0] * self.size[1] <= 1.6 * 1024 * 1024:
            self.prewarm(vid)
        return version

    def _crop_edit(self, parent, source, window, task, boxes, colors, regions, seed, feather_px, grow_px, vid, latent_id,
                   sigmas=None, attention=None, sdedit=False):
        """Generate only ``window`` and paste it back; returns (full image, result, latent id or None)."""
        x0, y0, x1, y1 = window
        cells = [x0 // 16, y0 // 16, x1 // 16, y1 // 16]
        local = [(bx0 - x0, by0 - y0, bx1 - x0, by1 - y0) for bx0, by0, bx1, by1 in boxes]
        crop_image = source.crop(window)
        crop_path = os.path.join(self.dir, f"{vid}_in.png")
        crop_image.save(crop_path, compress_level=1)
        mt: dict = {
            "prefix_cache": self.cache,
            **({"sigmas": list(sigmas)} if sigmas else {}),
            "mask": {"boxes": [list(b) for b in local], "canvas_ref": 0, "feather_px": feather_px,
                     "grow_px": grow_px, "annotate": self.mark_boxes or bool(regions), "init_from_source": sdedit},
        }
        if colors:
            mt["mask"]["colors"] = [list(c) for c in colors]
        if self.passthrough and parent.latent_id:
            # clean source = a window of the parent's latents; the result is written back into a copy
            mt["ref_latents"] = {0: parent.latent_id}
            mt["ref_latent_crops"] = {0: cells}
            mt["compose"] = {"base": parent.latent_id, "box": cells, "as": latent_id}
        else:
            latent_id = None  # no full latents to compose into; the next turn VAE-encodes this version
        out, result = self.engine.run(
            prompt=task, images=[crop_path], width=x1 - x0, height=y1 - y0, seed=seed, mt=mt,
            name=f"session_{self.id}_{vid}_crop", steps=self.steps, guidance_scale=self.guidance_scale,
            attention=attention,
        )
        out = out.convert(source.mode)
        patch = blend_box_edit(out, crop_image, local, feather_px, grow_px) if self.paste_back else out
        full = source.copy()
        full.paste(patch, (x0, y0))
        return full, result, latent_id

    def reroll(self, seed: int | None = None) -> Version:
        """Redo the head's edit from its parent with a new seed (its prefix K/V is fully cached)."""
        head = self.current
        if head.parent is None:
            raise ValueError("nothing to re-roll at the root")
        return self.edit(
            (head.request if head.request is not None else head.instruction) or "", boxes=None if head.regions else head.boxes, regions=head.regions, seed=seed,
            feather_px=head.feather_px, grow_px=head.grow_px, from_version=head.parent,
        )
