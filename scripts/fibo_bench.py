"""Fibo 2.0 Edit on SGLang: latency at 1K and 2K, full and masked, saving every output.

  python scripts/fibo_bench.py --image photo.jpg --out out/fibo_sgl [--no-graphs] [--attention sage_attn]
"""

import argparse
import json
import os
import time

from PIL import Image

EDITS = [
    ("full", "Make it night time with warm lamp light.", None),
    ("full", "Turn it into a watercolor painting.", None),
    ("box", "Add a small potted cactus.", (0.08, 0.55, 0.35, 0.9)),
    ("box", "Put a red coffee mug here.", (0.6, 0.6, 0.85, 0.9)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out/fibo_sgl")
    ap.add_argument("--sizes", default="1024,2048")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--attention", default="sage_attn")
    args = ap.parse_args()

    from multi_turn.app.session import Engine, _to_pil

    kw = {"component_attention_backends": {"transformer": args.attention}} if args.attention else {}
    if not args.no_graphs:
        kw["enable_breakable_cuda_graph"] = True
    engine = Engine("briaai/Fibo-2.0-edit-turbo", out_dir=args.out, **kw)
    os.makedirs(args.out, exist_ok=True)
    src = Image.open(args.image).convert("RGB")
    report = []
    for size in map(int, args.sizes.split(",")):
        scale = size / max(src.size)
        w, h = round(src.width * scale / 16) * 16, round(src.height * scale / 16) * 16
        path = os.path.join(args.out, f"src_{size}.png")
        src.resize((w, h), Image.LANCZOS).save(path)
        for i, (kind, prompt, frac) in enumerate(EDITS):
            fibo = {"mask_boxes": [[frac[0] * w, frac[1] * h, frac[2] * w, frac[3] * h]]} if frac else {}
            req = dict(prompt=prompt, images=[path], width=w, height=h, seed=0, mt={}, name=f"{size}_{i}")
            run = lambda: engine.generator.generate(sampling_params_kwargs=dict(
                prompt=prompt, image_path=[path], width=w, height=h, seed=0, save_output=False,
                diffusers_kwargs={"fibo": fibo}))
            run()
            times = []
            for _ in range(args.repeats):
                t = time.perf_counter()
                res = run()
                times.append(time.perf_counter() - t)
            res = res[0] if isinstance(res, list) else res
            _to_pil(res).save(os.path.join(args.out, f"{size}_{i}_{kind}.png"))
            row = dict(size=size, wh=(w, h), kind=kind, prompt=req["prompt"], seconds=min(times))
            print(row, flush=True)
            report.append(row)
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    engine.shutdown()


if __name__ == "__main__":
    main()
