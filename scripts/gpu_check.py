"""GPU validation for mtedit: correctness of every cache path, masks, passthrough, timings.

  MTEDIT_DEBUG=1 python scripts/gpu_check.py --model Qwen/Qwen-Image-2.1-Turbo --image chess_test/board_clean.png

Checks (same seed throughout):
  miss     first cached request == uncached request (store insert must not change compute)
  hit      identical request again: full prefix from the store, == uncached
  partial  same image, new prompt: header + image from the store, prompt prefilled by mtedit
  mask     boxed edit: raw output outside the boxes vs the source
  chain    3 passthrough turns vs 3 VAE re-encode turns (drift of untouched area)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), np.float64)
    y = np.asarray(b.convert("RGB").resize(a.size), np.float64)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def identical(a: Image.Image, b: Image.Image) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def outside_diff(out: Image.Image, src: Image.Image, boxes, margin=24):
    x = np.asarray(out.convert("RGB"), np.float64)
    y = np.asarray(src.convert("RGB").resize(out.size), np.float64)
    keep = np.ones(x.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        keep[max(0, int(y0) - margin) : int(y1) + margin, max(0, int(x0) - margin) : int(x1) + margin] = False
    d = np.abs(x - y)[keep]
    return float(d.mean()), float(d.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1-Turbo")
    ap.add_argument("--image", default="chess_test/board_clean.png")
    ap.add_argument("--out", default="/workspace/mt/check")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--int8", action="store_true", help="online INT8 ConvRot DiT (needs CUDA_HOME at CUDA 13)")
    args = ap.parse_args()

    from mtedit.session import Engine, EditSession

    os.makedirs(args.out, exist_ok=True)
    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if args.int8 else {}
    engine = Engine(args.model, out_dir=args.out, **extra)
    size = args.size
    src_path = os.path.join(args.out, "src.png")
    Image.open(args.image).convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(src_path)
    src = Image.open(src_path)
    report = {}

    def run(tag, prompt, mt, seed=42, images=None):
        t = time.perf_counter()
        img, res = engine.run(prompt=prompt, images=images or [src_path], width=size, height=size,
                              seed=seed, mt=mt, name=tag)
        dt = time.perf_counter() - t
        img.load()
        print(f"[{tag}] {dt:.3f}s", flush=True)
        return img, dt

    p1 = "Move the white pawn from e2 to e4. Keep every other square and piece exactly unchanged."
    p2 = "Move the white knight from g1 to f3. Keep every other square and piece exactly unchanged."

    # warm up kernels/allocator so timings compare steady state
    run("warmup", p1, {"prefix_cache": False}, seed=1)

    base1, t_base1 = run("p1_nocache", p1, {"prefix_cache": False})
    miss1, t_miss1 = run("p1_miss", p1, {})
    hits = [run(f"p1_hit{i}", p1, {}) for i in range(args.repeats)]
    base2, t_base2 = run("p2_nocache", p2, {"prefix_cache": False})
    part2, t_part2 = run("p2_partial", p2, {})
    nocache_t = [run(f"p1_nocache{i}", p1, {"prefix_cache": False})[1] for i in range(args.repeats)]

    report["cache"] = dict(
        miss_identical=identical(miss1, base1),
        hit_identical=all(identical(h, base1) for h, _ in hits),
        partial_psnr_vs_nocache=psnr(part2, base2),
        partial_identical=identical(part2, base2),
        seconds=dict(
            nocache=float(np.median(nocache_t)),
            miss=t_miss1,
            full_hit=float(np.median([t for _, t in hits])),
            partial=t_part2,
            nocache_p2=t_base2,
        ),
    )

    # mask: e2 -> e4 boxes on a 1024 board (128 px squares, a1 bottom-left)
    sq = size / 8
    def box(square):
        f, r = "abcdefgh".index(square[0]), int(square[1])
        return (f * sq, (8 - r) * sq, (f + 1) * sq, (9 - r) * sq)
    boxes = [box("e2"), box("e4")]
    pm = ("Move the white pawn from the upper-left marked square region to the other: the pawn on e2 moves to e4, "
          "e2 becomes an empty light square. Keep every other square and piece exactly unchanged.")
    masked, t_mask = run("mask_e2e4", pm, {"mask": {"boxes": [list(b) for b in boxes], "canvas_ref": 0, "feather_px": 16}})
    unmasked, _ = run("nomask_e2e4", pm, {})
    report["mask"] = dict(
        outside_mean_max_masked=outside_diff(masked, src, boxes),
        outside_mean_max_unmasked=outside_diff(unmasked, src, boxes),
        seconds=t_mask,
    )

    # chain drift: passthrough vs VAE re-encode, paste-back off to see the raw model drift
    moves = [("e2", "e4", "white pawn"), ("e7", "e5", "black pawn"), ("g1", "f3", "white knight")]
    drift = {}
    for passthrough in (True, False):
        s = EditSession(engine, src_path, size=(size, size), passthrough=passthrough, paste_back=False)
        all_boxes = []
        for a, b, piece in moves:
            bx = [box(a), box(b)]
            all_boxes += bx
            s.edit(f"Move the {piece} from {a} to {b}; {a} becomes an empty square. Keep everything else exactly unchanged.",
                   boxes=bx, seed=7)
        drift["passthrough" if passthrough else "vae_reencode"] = dict(
            outside_mean_max=outside_diff(s.image(), src, all_boxes),
            final=s.path(),
            seconds=[v.seconds for v in s.lineage()[1:]],
        )
    report["chain"] = drift

    print(json.dumps(report, indent=2))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    engine.shutdown()


if __name__ == "__main__":
    main()
