"""Calibrate the change lock from MTEDIT_LOCK_DUMP files (one request per box_compare case, in order)."""
import glob
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/workspace/multi_turn/scripts/demo")
from box_compare import CASES, OUT  # noqa: E402

files = sorted(glob.glob("/workspace/mt/dump/*.npz"))
reqs, cur = [], []
for f in files:
    z = np.load(f)
    step = int(f.rsplit("_s", 1)[1].split(".")[0])
    if step == 0 and cur:
        reqs.append(cur); cur = []
    cur.append(z)
reqs.append(cur)
print("requests", len(reqs), [len(r) for r in reqs])


def pool(d, k):
    t = torch.from_numpy(d).permute(2, 0, 1)[None]
    return F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="replicate"), k, 1)[0].permute(1, 2, 0).numpy()


for (name, _, _, regions), req in zip(CASES, reqs[-len(CASES):]):
    src = np.asarray(Image.open(f"{OUT}/src/{name}.png").convert("RGB"), np.float32)
    out = np.asarray(Image.open(f"{OUT}/z_dump/{name}.png").convert("RGB"), np.float32)
    inside = req[0]["inside"] > 0.5
    h, w = inside.shape
    # the request is a crop window: find it from the box (fractions of the full image) is not needed; compare
    # per-cell final pixel change on the crop grid by resizing the full diff to the crop's box region instead
    diff = np.abs(out - src).mean(-1)
    H, W = diff.shape
    bx = regions[0]["box"]
    # crop grid covers some window; locate the inside cells' bbox and map to the box in pixels
    ys, xs = np.where(inside)
    by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    px = diff[int(bx[1] * H):int(bx[3] * H), int(bx[0] * W):int(bx[2] * W)]
    truth = np.asarray(Image.fromarray(px).resize((bx1 - bx0, by1 - by0), Image.BOX), np.float32)
    print(f"\n== {name}: grid {h}x{w}, box cells {inside.sum()}, final cell diff q50 {np.median(truth):.1f} q90 {np.quantile(truth, .9):.1f}"
          f" obj(>12) {np.mean(truth > 12):.2f}")
    obj = truth > 12
    for z in req:
        outside = z["editable"] == 0
        for k in (1, 3, 5):
            d = np.sqrt((pool(z["d"], k) ** 2).mean(-1)) if k > 1 else np.sqrt((z["d"] ** 2).mean(-1))
            floor = np.quantile(d[outside], .9) if outside.sum() > 16 else np.nan
            din = d[by0:by1, bx0:bx1]
            o = din[obj].mean() if obj.any() else np.nan
            b = din[~obj].mean() if (~obj).any() else np.nan
            # best single threshold accuracy vs truth
            best = max(((np.mean((din > t) == obj), t) for t in np.linspace(din.min(), din.max(), 60)))
            print(f"  s={float(z['sigma']):.2f} pool{k}: floor {floor:.3f} obj {o:.3f} bg {b:.3f} best acc {best[0]:.2f} @ {best[1]:.3f} ({best[1] / floor:.2f}x floor)")
