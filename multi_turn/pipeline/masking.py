from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from multi_turn.request import MaskSpec

LATENT_STRIDE = 16


def _cell_centers(height: int, width: int):
    h, w = height // LATENT_STRIDE, width // LATENT_STRIDE
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) * LATENT_STRIDE
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) * LATENT_STRIDE
    return h, w, ys, xs


def rasterize_inside(spec: MaskSpec, height: int, width: int) -> torch.Tensor:
    h, w, ys, xs = _cell_centers(height, width)
    out = torch.zeros(h, w)
    for x0, y0, x1, y1 in spec.boxes:
        out = torch.maximum(out, (((ys >= y0) & (ys < y1))[:, None] & ((xs >= x0) & (xs < x1))[None, :]).float())
    return out.reshape(-1)


def rasterize(spec: MaskSpec, height: int, width: int) -> torch.Tensor:
    h, w, ys, xs = _cell_centers(height, width)
    feather = max(spec.feather_px, 1e-3)
    weight = torch.zeros(h, w)
    for x0, y0, x1, y1 in spec.boxes:
        x0, y0, x1, y1 = x0 - spec.grow_px, y0 - spec.grow_px, x1 + spec.grow_px, y1 + spec.grow_px
        # feathered outward only: an inward ramp leaves remnants of removed objects in tight boxes
        dy = torch.maximum(y0 - ys, ys - y1).clamp(min=0)[:, None]
        dx = torch.maximum(x0 - xs, xs - x1).clamp(min=0)[None, :]
        dist = torch.sqrt(dx * dx + dy * dy)
        weight = torch.maximum(weight, (1 - dist / feather).clamp(min=0))
    return weight.reshape(-1)


@dataclass
class MaskState:
    x0: torch.Tensor
    editable: torch.Tensor
    eps: torch.Tensor | None = None
    init_from_source: bool = False
    inside: torch.Tensor | None = None
    hw: tuple[int, int] = (0, 0)
    lock_sigma: float = 0.0
    locked: bool = False
    sigmas_host: list | None = None

    def lock(self, x0_pred: torch.Tensor, ratio: float = 2.0, min_floor: float = 0.05) -> dict:
        """Freeze box cells the edit did not change. The floor is the 90th-percentile prediction drift
        outside the box; binary, because a partial mix reads as a ghost of the previous version."""
        self.locked = True
        h, w = self.hw
        x0 = self.x0.to(x0_pred)
        d = (x0_pred[:1].float() - x0.float()).pow(2).mean(-1).sqrt().view(h, w)
        inside = self.inside.to(d.device).view(h, w) > 0.5
        outside = self.editable.to(d.device).view(h, w) == 0
        if outside.sum() < 16 or inside.sum() < 4:
            return {"skipped": "no outside reference"}
        floor = torch.quantile(d[outside], 0.9).clamp(min=min_floor)
        # calibrated on 5 box edits at sigma 0.70: the object/background split sits at 1.95-3.25x floor
        thr = floor * ratio
        hard = (d > thr).float()
        frac = hard[inside].mean()
        if frac < 0.01:
            return {"skipped": "no change", "floor": float(floor)}
        if frac > 0.7:
            return {"skipped": "restyle", "changed": float(frac)}
        c = F.max_pool2d(hard[None, None], 5, 1, 2)
        c = torch.maximum(c, F.avg_pool2d(F.pad(c, (1, 1, 1, 1), mode="replicate"), 3, 1))
        c = c.view(1, -1, 1)
        ins = self.inside.to(c)
        e = self.editable.to(c)
        self.editable = (e * (1 - ins) + ins * c).to(self.editable.dtype)
        return {"floor": float(floor), "thr": float(thr), "changed": float(frac)}

    def blend(self, latents: torch.Tensor, sigma) -> torch.Tensor:
        x0 = self.x0.to(device=latents.device, dtype=latents.dtype)
        eps = self.eps.to(device=latents.device, dtype=latents.dtype)
        m = self.editable.to(device=latents.device, dtype=latents.dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=latents.device, dtype=latents.dtype)
        keep = (1 - sigma) * x0 + sigma * eps
        return m * latents + (1 - m) * keep
