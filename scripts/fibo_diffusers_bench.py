"""Fibo 2.0 Edit Turbo: edit quality and latency at 1K and 2K, full and masked.

  python scripts/fibo_diffusers_bench.py --image photo.jpg --out out/fibo   # env with Bria's diffusers branch
"""

import argparse
import json
import os
import time

import torch
from diffusers import BriaFibo2Pipeline
from PIL import Image, ImageDraw

EDITS = [
    ("full", "Make it night time with warm lamp light.", None),
    ("full", "Turn it into a watercolor painting.", None),
    ("box", "Add a small potted cactus.", (0.08, 0.55, 0.35, 0.9)),
    ("box", "Put a red coffee mug here.", (0.6, 0.6, 0.85, 0.9)),
]


def mask_for(size, frac):
    m = Image.new("L", size, 0)
    w, h = size
    ImageDraw.Draw(m).rectangle([frac[0] * w, frac[1] * h, frac[2] * w, frac[3] * h], fill=255)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out/fibo")
    ap.add_argument("--sizes", default="1024,2048")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-edit-turbo", dtype=torch.bfloat16).to("cuda")
    src = Image.open(args.image).convert("RGB")
    report = []
    for size in map(int, args.sizes.split(",")):
        scale = size / max(src.size)
        image = src.resize((round(src.width * scale / 16) * 16, round(src.height * scale / 16) * 16), Image.LANCZOS)
        image.save(f"{args.out}/src_{size}.png")
        for i, (kind, prompt, frac) in enumerate(EDITS):
            kw = dict(prompt=prompt, image=image, width=image.width, height=image.height,
                      num_inference_steps=4, guidance_scale=1.0,
                      generator=torch.Generator("cuda").manual_seed(0))
            if frac:
                kw["mask"] = mask_for(image.size, frac)
            pipe(**kw)  # warm-up per shape
            times = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = pipe(**kw).images[0]
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            out.save(f"{args.out}/{size}_{i}_{kind}.png")
            row = dict(size=size, wh=image.size, kind=kind, prompt=prompt, seconds=min(times))
            print(row, flush=True)
            report.append(row)
    print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    json.dump(report, open(f"{args.out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
