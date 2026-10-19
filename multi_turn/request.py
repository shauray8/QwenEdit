from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PAYLOAD_KEY = "mt"


@dataclass
class MaskSpec:
    boxes: list[tuple[float, float, float, float]]
    canvas_ref: int = 0
    feather_px: float = 16.0
    grow_px: float = 0.0
    annotate: bool = False
    mark_color: tuple[int, int, int] = (255, 0, 0)
    mark_width_px: float = 6.0
    colors: list[tuple[int, int, int]] | None = None
    init_from_source: bool = False
    # change lock: from this sigma on, unchanged cells inside the box are frozen to the source. 0 disables
    lock_sigma: float = 0.75


@dataclass
class MTRequest:
    store_output_as: str | None = None
    ref_latents: dict[int, str] = field(default_factory=dict)
    mask: MaskSpec | None = None
    prefix_cache: bool = True
    prefill_only: bool = False
    ref_latent_crops: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    compose: dict | None = None
    store_ref_as: dict[int, str] = field(default_factory=dict)
    sigmas: list[float] | None = None
    ref_scale: float = 1.0
    progressive: int = 0
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
                lock_sigma=float(mask.get("lock_sigma", 0.75)),
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
