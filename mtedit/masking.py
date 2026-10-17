"""Latent-mask region edits for flow matching.

After every scheduler step, latents outside the editable region are replaced by
the source latents noised to the next sigma, ``(1 - s) * x0 + s * eps``, with the
request's own initial noise as ``eps``. At the final sigma (0) the outside is
exactly the source latents, so repeated edits never drift outside their boxes
in latent space; the session's pixel paste-back removes the remaining VAE
decode differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mtedit.request import MaskSpec

LATENT_STRIDE = 16  # Qwen-Image 2.1 VAE spatial compression


def rasterize_inside(spec: MaskSpec, height: int, width: int) -> torch.Tensor:
    """1 for latent cells whose center is inside a box, ``[h * w]``."""
    h, w = height // LATENT_STRIDE, width // LATENT_STRIDE
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) * LATENT_STRIDE
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) * LATENT_STRIDE
    out = torch.zeros(h, w)
    for x0, y0, x1, y1 in spec.boxes:
        out = torch.maximum(out, (((ys >= y0) & (ys < y1))[:, None] & ((xs >= x0) & (xs < x1))[None, :]).float())
    return out.reshape(-1)


def rasterize(spec: MaskSpec, height: int, width: int) -> torch.Tensor:
    """Editable weight per latent cell, ``[h * w]`` in [0, 1] (1 = regenerate)."""
    h, w = height // LATENT_STRIDE, width // LATENT_STRIDE
    # cell centers in target pixels
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) * LATENT_STRIDE
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) * LATENT_STRIDE
    feather = max(spec.feather_px, 1e-3)
    weight = torch.zeros(h, w)
    for x0, y0, x1, y1 in spec.boxes:
        x0, y0, x1, y1 = x0 - spec.grow_px, y0 - spec.grow_px, x1 + spec.grow_px, y1 + spec.grow_px
        # fully editable inside the box, feathered outward only: an inward ramp pulls the original back
        # into the box edge, leaving remnants of removed objects in tight boxes
        dy = torch.maximum(y0 - ys, ys - y1).clamp(min=0)[:, None]
        dx = torch.maximum(x0 - xs, xs - x1).clamp(min=0)[None, :]
        dist = torch.sqrt(dx * dx + dy * dy)
        weight = torch.maximum(weight, (1 - dist / feather).clamp(min=0))
    return weight.reshape(-1)


@dataclass
class MaskState:
    x0: torch.Tensor  # [1, HW, C] source latents (normalized, packed)
    editable: torch.Tensor  # [1, HW, 1]
    eps: torch.Tensor | None = None
    init_from_source: bool = False
    inside: torch.Tensor | None = None  # [1, HW, 1] cells strictly inside a box (no feather)
    hw: tuple[int, int] = (0, 0)  # latent grid
    lock_sigma: float = 0.0
    locked: bool = False
    sigmas_host: list | None = None

    def lock(self, x0_pred: torch.Tensor, lo: float = 2.0, hi: float = 3.0, min_floor: float = 0.05) -> dict:
        """Freeze box cells whose predicted clean latent still matches the source.

        The threshold is the model's own re-render error: outside the box the latents are the
        noised source, so ``|x0_pred - x0|`` there is pure prediction drift. Inside the box,
        cells under 2-3x that drift (its 90th percentile) were only re-rendered and go
        back to the source for the remaining steps; changed cells stay editable, dilated by one
        cell and softened, so the new content finishes against the exact background.
        """
        import torch.nn.functional as F

        self.locked = True
        h, w = self.hw
        x0 = self.x0.to(x0_pred)
        d = (x0_pred[:1].float() - x0.float()).pow(2).mean(-1).sqrt().view(h, w)  # [h, w]
        inside = self.inside.to(d.device).view(h, w) > 0.5
        outside = self.editable.to(d.device).view(h, w) == 0
        if outside.sum() < 16 or inside.sum() < 4:
            return {"skipped": "no outside reference"}
        floor = torch.quantile(d[outside], 0.9).clamp(min=min_floor)
        # calibrated on 5 box edits at Turbo sigma 0.70: the best object/background split is 1.95-3.25x the
        # outside drift (88-93% of cells right). Binary at 2x (the generous end): a changed cell is fully
        # regenerated, never a mix with the source, which reads as a soft ghost of the previous version
        thr = floor * lo
        hard = (d > thr).float()
        frac = hard[inside].mean()
        if frac < 0.01:  # nothing clearly changed yet: do not erase a subtle edit
            return {"skipped": "no change", "floor": float(floor)}
        if frac > 0.7:  # a restyle of the whole box: lock nothing
            return {"skipped": "restyle", "changed": float(frac)}
        c = F.max_pool2d(hard[None, None], 5, 1, 2)  # two cells (32 px) fully kept around every change
        # soften only outward: the blur may add weight beyond the kept cells, never remove it inside them
        c = torch.maximum(c, F.avg_pool2d(F.pad(c, (1, 1, 1, 1), mode="replicate"), 3, 1))
        c = c.view(1, -1, 1)
        ins = self.inside.to(c)
        e = self.editable.to(c)
        # inside the box: editable only where changed; the outward feather keeps its own weight
        self.editable = (e * (1 - ins) + ins * c).to(self.editable.dtype)
        return {"floor": float(floor), "thr": float(thr), "changed": float(frac)}

    def blend(self, latents: torch.Tensor, sigma) -> torch.Tensor:
        """``sigma`` is a float or a 0-d tensor (kept on device to avoid a host sync)."""
        x0 = self.x0.to(device=latents.device, dtype=latents.dtype)
        eps = self.eps.to(device=latents.device, dtype=latents.dtype)
        m = self.editable.to(device=latents.device, dtype=latents.dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=latents.device, dtype=latents.dtype)
        keep = (1 - sigma) * x0 + sigma * eps
        return m * latents + (1 - m) * keep
