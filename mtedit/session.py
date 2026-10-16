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
import time
import uuid
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter

Box = tuple[float, float, float, float]

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
    feather_px: float = 16.0
    grow_px: float = 0.0
    seed: int | None = None
    latent_id: str | None = None  # worker-side final latents, for passthrough
    seconds: float = 0.0
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

    def run(self, *, prompt, images, width, height, seed, mt, name, steps=None, guidance_scale=None):
        kwargs = dict(
            prompt=prompt,
            image_path=list(images),
            width=width,
            height=height,
            seed=seed,
            save_output=True,
            output_path=self.out_dir,
            output_file_name=name,
            diffusers_kwargs={"mt": mt},
        )
        if steps is not None:
            kwargs["num_inference_steps"] = steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        result = self.generator.generate(sampling_params_kwargs=kwargs)
        result = result[0] if isinstance(result, list) else result
        if result is None:
            raise RuntimeError("generation failed")
        return Image.open(result.output_file_path), result

    def shutdown(self):
        self.generator.shutdown()


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
        root.save(path)
        self.versions: dict[str, Version] = {"v000": Version(id="v000", parent=None, image_path=path)}
        self.root = self.head = "v000"
        self._counter = 0

    # history ---------------------------------------------------------------

    @property
    def current(self) -> Version:
        return self.versions[self.head]

    def image(self, version: str | None = None) -> Image.Image:
        return Image.open(self.versions[version or self.head].image_path)

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
        instruction: str,
        *,
        boxes: list[Box] | None = None,
        seed: int | None = None,
        feather_px: float = 16.0,
        grow_px: float = 0.0,
        from_version: str | None = None,
    ) -> Version:
        """Apply one edit to ``from_version`` (default: head) and make the result the head."""
        parent = self.versions[from_version or self.head]
        refs = [parent]
        if self.layout == "anchor" and parent.id != self.root:
            refs = [self.versions[self.root], parent]
        canvas_ref = len(refs) - 1
        template = self.template if len(refs) == 2 else "{instruction}"
        prompt = template.format(instruction=instruction.strip())

        self._counter += 1
        vid = f"v{self._counter:03d}"
        latent_id = f"{self.id}/{vid}"
        mt: dict = {"store_output_as": latent_id}
        if self.passthrough:
            mt["ref_latents"] = {i: r.latent_id for i, r in enumerate(refs) if r.latent_id}
        if boxes:
            mt["mask"] = {"boxes": [list(b) for b in boxes], "canvas_ref": canvas_ref, "feather_px": feather_px, "grow_px": grow_px}
        seed = int(time.time_ns() % (2**31)) if seed is None else seed

        start = time.perf_counter()
        out, _ = self.engine.run(
            prompt=prompt,
            images=[r.image_path for r in refs],
            width=self.size[0],
            height=self.size[1],
            seed=seed,
            mt=mt,
            name=f"session_{self.id}_{vid}_raw",
            steps=self.steps,
            guidance_scale=self.guidance_scale,
        )
        seconds = time.perf_counter() - start
        out = out.convert(Image.open(parent.image_path).mode)
        if boxes and self.paste_back:
            source = Image.open(parent.image_path)
            out = Image.composite(out, source, pixel_mask(out.size, boxes, feather_px, grow_px))
        path = os.path.join(self.dir, f"{vid}.png")
        out.save(path)

        version = Version(
            id=vid, parent=parent.id, image_path=path, instruction=instruction,
            boxes=boxes, feather_px=feather_px, grow_px=grow_px, seed=seed,
            latent_id=latent_id, seconds=seconds,
        )
        self.versions[vid] = version
        parent.children.append(vid)
        self.head = vid
        return version

    def reroll(self, seed: int | None = None) -> Version:
        """Redo the head's edit from its parent with a new seed (its prefix K/V is fully cached)."""
        head = self.current
        if head.parent is None:
            raise ValueError("nothing to re-roll at the root")
        return self.edit(
            head.instruction, boxes=head.boxes, seed=seed, feather_px=head.feather_px,
            grow_px=head.grow_px, from_version=head.parent,
        )
