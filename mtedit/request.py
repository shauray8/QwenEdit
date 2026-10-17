"""Per-request multi-turn payload, carried in ``diffusers_kwargs["mt"]``.

``diffusers_kwargs`` already travels from both ``DiffGenerator.generate`` and the
HTTP ``/v1/images/generations`` endpoint into ``batch.extra``, so no new
request fields are needed. Requests without a payload still get automatic
prefix reuse; the payload only adds passthrough, masks and output storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PAYLOAD_KEY = "mt"


@dataclass
class MaskSpec:
    # (x0, y0, x1, y1) in target pixels; the union is editable, the rest is kept
    boxes: list[tuple[float, float, float, float]]
    # reference whose latents are the "keep" source; must match the target size
    canvas_ref: int = 0
    # soft edge width in target pixels (converted to latent cells)
    feather_px: float = 16.0
    # grow each box by this many target pixels before rasterizing
    grow_px: float = 0.0
    # draw the boxes onto the canvas reference the model sees (Qwen-Image 2.1's native
    # mark-guided local editing); the kept outside still comes from the clean latents
    annotate: bool = False
    mark_color: tuple[int, int, int] = (255, 0, 0)
    mark_width_px: float = 6.0
    # optional per-box outline colors (multi-region edits name each box by its color)
    colors: list[tuple[int, int, int]] | None = None
    # modify edits: start the whole latent from the source noised to the schedule's first sigma
    # (SDEdit) instead of pure noise, so geometry inside the box stays aligned with the outside
    init_from_source: bool = False


@dataclass
class MTRequest:
    # store this request's final latents under this id (and "<id>#<k>" for k > 0)
    store_output_as: str | None = None
    # reference index -> latent id from a previous ``store_output_as``
    ref_latents: dict[int, str] = field(default_factory=dict)
    mask: MaskSpec | None = None
    prefix_cache: bool = True
    # warm the prefix pools only: run the first denoising call (the prefill) and skip the rest
    prefill_only: bool = False
    # reference index -> (x0, y0, x1, y1) in latent cells: use that window of its stored latents
    ref_latent_crops: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    # crop edits: write this request's latents into a copy of ``base``'s full latents at ``box``
    # (latent cells) and store the result as ``as``
    compose: dict | None = None
    # reference index -> latent id: keep that reference's clean condition latents for later passthrough
    store_ref_as: dict[int, str] = field(default_factory=dict)
    # explicit sigma schedule (e.g. a subset of Turbo's 8 distilled sigmas for a faster draft)
    sigmas: list[float] | None = None
    # reference area relative to the target (e.g. 0.25: a 1K reference for a 2K output); unboxed edits only
    ref_scale: float = 1.0
    # progressive resolution: the first ``progressive`` steps run at half width/height (unmasked edits only)
    progressive: int = 0
    # reference index -> area relative to the target, overriding ref_scale (e.g. small object references)
    ref_scales: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MTRequest":
        mask = data.get("mask")
        return cls(
            store_output_as=data.get("store_output_as"),
            ref_latents={int(k): str(v) for k, v in (data.get("ref_latents") or {}).items()},
            mask=MaskSpec(
                boxes=[tuple(map(float, b)) for b in mask["boxes"]],
                canvas_ref=int(mask.get("canvas_ref", 0)),
                feather_px=float(mask.get("feather_px", 16.0)),
                grow_px=float(mask.get("grow_px", 0.0)),
                annotate=bool(mask.get("annotate", False)),
                mark_color=tuple(int(c) for c in mask.get("mark_color", (255, 0, 0))),
                mark_width_px=float(mask.get("mark_width_px", 6.0)),
                colors=[tuple(int(c) for c in rgb) for rgb in mask["colors"]] if mask.get("colors") else None,
                init_from_source=bool(mask.get("init_from_source", False)),
            )
            if mask
            else None,
            prefix_cache=bool(data.get("prefix_cache", True)),
            prefill_only=bool(data.get("prefill_only", False)),
            ref_latent_crops={int(k): tuple(int(x) for x in v) for k, v in (data.get("ref_latent_crops") or {}).items()},
            compose=data.get("compose"),
            store_ref_as={int(k): str(v) for k, v in (data.get("store_ref_as") or {}).items()},
            sigmas=[float(x) for x in data["sigmas"]] if data.get("sigmas") else None,
            ref_scale=float(data.get("ref_scale", 1.0)),
            progressive=int(data.get("progressive", 0)),
            ref_scales={int(k): float(v) for k, v in (data.get("ref_scales") or {}).items()},
        )

    @classmethod
    def from_batch(cls, batch) -> "MTRequest":
        kwargs = (getattr(batch, "extra", None) or {}).get("diffusers_kwargs") or {}
        data = kwargs.get(PAYLOAD_KEY)
        return cls.from_dict(data) if isinstance(data, dict) else cls()
