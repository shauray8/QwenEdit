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
    children: list[str] = field(default_factory=list)


class Engine:
    """One ``DiffGenerator`` (and its GPU worker) shared by any number of sessions."""

    def __init__(self, model_path: str = "Qwen/Qwen-Image-2.1-Turbo", out_dir: str = "mt_out", **server_kwargs):
        from sglang.multimodal_gen import DiffGenerator

        server_kwargs.setdefault("performance_mode", "speed")
        self.model_path = model_path
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.generator = DiffGenerator.from_pretrained(model_path=model_path, **server_kwargs)
        self.lock = threading.Lock()  # one generation at a time (edits and background warm-ups)

    def run(self, *, prompt, images, width, height, seed, mt, name, steps=None, guidance_scale=None, save=False):
        """One generation; returns (PIL image, result). ``save=False`` keeps the output in memory."""
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


def _round32(value: float) -> int:
    return max(32, int(round(value / 32)) * 32)


def pixel_mask(size: tuple[int, int], boxes: list[Box], feather_px: float, grow_px: float) -> Image.Image:
    """Feathered editable mask in pixels (255 = take the new image)."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    pad = grow_px + feather_px / 2
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
        self.auto_prewarm = prewarm  # warm the next turn's prefix in the background after each edit
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mtedit-{session_id or 'io'}")
        self._saves: dict[str, Future] = {}
        self._warm: Future | None = None
        self.id = session_id or uuid.uuid4().hex[:8]
        self.dir = os.path.join(engine.out_dir, f"session_{self.id}")
        os.makedirs(self.dir, exist_ok=True)

        image = base_image if isinstance(base_image, Image.Image) else Image.open(base_image)
        if size is None:  # keep the aspect, ~1 MP, multiples of 32
            scale = (1024 * 1024 / (image.width * image.height)) ** 0.5
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

        def run():
            self.engine.run(
                prompt=template.format(instruction=""), images=[self._file(r.id) for r in refs],
                width=self.size[0], height=self.size[1], seed=0, mt=mt, name=f"session_{self.id}_warm",
                steps=self.steps, guidance_scale=self.guidance_scale,
            )

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
    ) -> Version:
        """Apply one edit to ``from_version`` (default: head) and make the result the head."""
        parent = self.versions[from_version or self.head]
        refs = self._refs(parent)
        canvas_ref = len(refs) - 1
        template = self.template if len(refs) == 2 else "{instruction}"
        instruction = (instruction or "").strip()
        colors = None
        if regions:  # per-box instructions: one outline color per box, named in the prompt
            if len(regions) > len(MARK_PALETTE):
                raise ValueError(f"at most {len(MARK_PALETTE)} regions per edit")
            boxes = [tuple(map(float, box)) for box, _ in regions]
            parts = [f"Inside the {MARK_PALETTE[i][0]} box: {text.strip().rstrip('.')}." for i, (_, text) in enumerate(regions)]
            colors = [MARK_PALETTE[i][1] for i in range(len(regions))]
            task = REGION_TEMPLATE.format(regions=" ".join(parts) + " ", instruction=instruction).replace("  ", " ")
        elif boxes and self.mark_boxes:
            task = self.box_template.format(instruction=instruction)
        else:
            task = instruction
        prompt = template.format(instruction=task)

        self._counter += 1
        vid = f"v{self._counter:03d}"
        latent_id = f"{self.id}/{vid}"
        mt: dict = {"store_output_as": latent_id, "prefix_cache": self.cache}
        if self.passthrough:
            mt["ref_latents"] = {i: r.latent_id for i, r in enumerate(refs) if r.latent_id}
        if boxes:
            mt["mask"] = {
                "boxes": [list(b) for b in boxes], "canvas_ref": canvas_ref, "feather_px": feather_px,
                "grow_px": grow_px, "annotate": self.mark_boxes or bool(regions),
            }
            if colors:
                mt["mask"]["colors"] = [list(c) for c in colors]
        seed = int(time.time_ns() % (2**31)) if seed is None else seed

        start = time.perf_counter()
        out, result = self.engine.run(
            prompt=prompt,
            images=[self._file(r.id) for r in refs],
            width=self.size[0],
            height=self.size[1],
            seed=seed,
            mt=mt,
            name=f"session_{self.id}_{vid}_raw",
            steps=self.steps,
            guidance_scale=self.guidance_scale,
        )
        generated = time.perf_counter()
        source = self._image_cache.get(parent.id) or Image.open(self.path(parent.id))
        out = out.convert(source.mode)
        if boxes and self.paste_back:
            out = Image.composite(out, source, pixel_mask(out.size, boxes, feather_px, grow_px))
        posted = time.perf_counter()
        path = os.path.join(self.dir, f"{vid}.png")
        self._image_cache[vid] = out
        # the next turn reads the file; level 1 is ~10x faster than the default and it is off the critical path
        self._saves[vid] = self._io.submit(out.save, path, compress_level=1)
        seconds = time.perf_counter() - start
        timings = dict(generate=generated - start, post=posted - generated, save=time.perf_counter() - posted,
                       server=getattr(result, "generation_time", 0.0))

        version = Version(
            id=vid, parent=parent.id, image_path=path, instruction=instruction, regions=regions,
            boxes=boxes, feather_px=feather_px, grow_px=grow_px, seed=seed,
            latent_id=latent_id, seconds=seconds, timings=timings,
        )
        self.versions[vid] = version
        parent.children.append(vid)
        self.head = vid
        if self.auto_prewarm:
            self.prewarm(vid)
        return version

    def reroll(self, seed: int | None = None) -> Version:
        """Redo the head's edit from its parent with a new seed (its prefix K/V is fully cached)."""
        head = self.current
        if head.parent is None:
            raise ValueError("nothing to re-roll at the root")
        return self.edit(
            head.instruction or "", boxes=None if head.regions else head.boxes, regions=head.regions, seed=seed,
            feather_px=head.feather_px, grow_px=head.grow_px, from_version=head.parent,
        )
