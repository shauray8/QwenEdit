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


@dataclass
class MTRequest:
    # store this request's final latents under this id (and "<id>#<k>" for k > 0)
    store_output_as: str | None = None
    # reference index -> latent id from a previous ``store_output_as``
    ref_latents: dict[int, str] = field(default_factory=dict)
    mask: MaskSpec | None = None
    prefix_cache: bool = True

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
            )
            if mask
            else None,
            prefix_cache=bool(data.get("prefix_cache", True)),
        )

    @classmethod
    def from_batch(cls, batch) -> "MTRequest":
        kwargs = (getattr(batch, "extra", None) or {}).get("diffusers_kwargs") or {}
        data = kwargs.get(PAYLOAD_KEY)
        return cls.from_dict(data) if isinstance(data, dict) else cls()
