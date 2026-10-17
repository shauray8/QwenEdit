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
        # signed distance to the box edge (negative inside); the ramp is centered on the edge so it
        # feathers inward as well as outward: a slightly shifted object blends instead of seaming
        dy = torch.maximum(y0 - ys, ys - y1)[:, None]
        dx = torch.maximum(x0 - xs, xs - x1)[None, :]
        outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2)
        signed = torch.where((dx <= 0) & (dy <= 0), torch.maximum(dx, dy), outside)
        weight = torch.maximum(weight, (0.5 - signed / feather).clamp(0, 1))
    return weight.reshape(-1)


@dataclass
class MaskState:
    x0: torch.Tensor  # [1, HW, C] source latents (normalized, packed)
    editable: torch.Tensor  # [1, HW, 1]
    eps: torch.Tensor | None = None
    init_from_source: bool = False

    def blend(self, latents: torch.Tensor, sigma) -> torch.Tensor:
        """``sigma`` is a float or a 0-d tensor (kept on device to avoid a host sync)."""
        x0 = self.x0.to(device=latents.device, dtype=latents.dtype)
        eps = self.eps.to(device=latents.device, dtype=latents.dtype)
        m = self.editable.to(device=latents.device, dtype=latents.dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=latents.device, dtype=latents.dtype)
        keep = (1 - sigma) * x0 + sigma * eps
        return m * latents + (1 - m) * keep
