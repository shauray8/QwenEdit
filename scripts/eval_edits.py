"""Edit quality over a fixed suite and several seeds: did the edit happen, and was the rest left alone.

  python scripts/eval_edits.py --image photo.jpg [--image other.jpg] --model fibo --seeds 4 --out out/eval_fibo
  python scripts/eval_edits.py --score out/eval_fibo        # rescore saved outputs

Metrics per output, averaged over seeds and edits:
  clip_t    CLIP ViT-L/14 cosine of the edited region (the box, or the whole image) with the edit's target text
  bg_lpips  LPIPS (AlexNet) outside the box, dilated 32 px, against the source; box edits only, lower is better
  bg_psnr   PSNR outside the dilated box against the source; box edits only
  clip_i    CLIP image cosine of a full-image edit with its source: how much of the scene survived
Compare configurations (attention backend, compile, pruning) on the same suite and seeds; one seed's PSNR only says
"a different sample" for a few-step distilled model.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

MODELS = {"qwen": "Qwen/Qwen-Image-2.1-Turbo", "fibo": "briaai/Fibo-2.0-edit-turbo"}
# (instruction, box as fractions of the image or None, CLIP target text or None to skip clip_t)
SUITE = [
    ("Make it night time with warm lamp light.", None, "a photo at night lit by warm lamps"),
    ("Turn it into a watercolor painting.", None, "a watercolor painting"),
    ("Make it a snowy winter day.", None, "a snowy winter scene"),
    ("Make the lighting golden hour.", None, "a photo in golden hour sunlight"),
    ("Add a small potted cactus.", (0.08, 0.55, 0.35, 0.9), "a small potted cactus"),
    ("Put a red coffee mug here.", (0.6, 0.6, 0.85, 0.9), "a red coffee mug"),
    ("Make it blue.", (0.6, 0.6, 0.85, 0.9), "a blue object"),
    ("Remove it.", (0.6, 0.6, 0.85, 0.9), None),
]
DILATE_PX = 32


def generate(args):
    if args.quant == "nvfp4":
        os.environ["MULTI_TURN_NVFP4"] = "1"
    from multi_turn.app.session import EditSession, Engine

    kw = {"component_attention_backends": {"transformer": args.attention}} if args.attention else {}
    engine = Engine(MODELS[args.model], out_dir=args.out, **kw)
    rows = []
    for k, path in enumerate(args.image):
        session = EditSession(engine, Image.open(path), megapixels=args.megapixels, session_id=f"eval{k}")
        W, H = session.size
        source = os.path.join(args.out, f"src{k}.png")
        session.image("v000").save(source)
        for i, (instruction, frac, target) in enumerate(SUITE):
            box = [frac[0] * W, frac[1] * H, frac[2] * W, frac[3] * H] if frac else None
            for seed in range(args.seeds):
                version = session.edit(instruction, boxes=[tuple(box)] if box else None, from_version="v000", seed=seed)
                out = os.path.join(args.out, f"{k}_{i}_{seed}.png")
                session.image(version.id).save(out)
                rows.append(dict(source=source, output=out, edit=i, seed=seed, instruction=instruction, box=box,
                                 target=target, seconds=version.seconds))
        session.flush()
    engine.shutdown()
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(rows, f, indent=1)


def score(out_dir):
    import lpips
    import torch
    from transformers import CLIPModel, CLIPProcessor

    rows = json.load(open(os.path.join(out_dir, "manifest.json")))
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval().half()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    lp = lpips.LPIPS(net="alex", verbose=False).cuda()

    def features(out):
        # transformers 5 returns the projected embedding as pooler_output, 4.x a bare tensor
        return torch.nn.functional.normalize((out if torch.is_tensor(out) else out.pooler_output).float(), dim=-1)

    @torch.no_grad()
    def embed_images(images):
        pix = proc(images=images, return_tensors="pt")["pixel_values"].cuda().half()
        return features(clip.get_image_features(pixel_values=pix))

    @torch.no_grad()
    def embed_text(text):
        tok = proc(text=[text], return_tensors="pt", padding=True).to("cuda")
        return features(clip.get_text_features(**tok))

    for r in rows:
        src, out = Image.open(r["source"]).convert("RGB"), Image.open(r["output"]).convert("RGB")
        region = out.crop(tuple(int(v) for v in r["box"])) if r["box"] else out
        if r["target"]:
            r["clip_t"] = float(embed_images([region]) @ embed_text(r["target"]).T)
        if r["box"]:
            W, H = out.size
            x0, y0, x1, y1 = r["box"]
            keep = np.ones((H, W), bool)
            keep[max(0, int(y0) - DILATE_PX):int(y1) + DILATE_PX, max(0, int(x0) - DILATE_PX):int(x1) + DILATE_PX] = False
            a, b = np.asarray(src, np.float32), np.asarray(out, np.float32)
            mse = ((a - b) ** 2)[keep].mean()
            r["bg_psnr"] = float(99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse))
            m = torch.from_numpy(keep).cuda()[None, None]
            ta = torch.from_numpy(a).permute(2, 0, 1)[None].cuda() / 127.5 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None].cuda() / 127.5 - 1
            with torch.no_grad():
                r["bg_lpips"] = float(lp(ta * m, tb * m))
        else:
            r["clip_i"] = float(embed_images([out]) @ embed_images([src]).T)

    summary = {}
    for key in ("clip_t", "bg_lpips", "bg_psnr", "clip_i", "seconds"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            summary[key] = round(float(np.mean(vals)), 4)
    summary["n"] = len(rows)
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append")
    ap.add_argument("--model", default="fibo", choices=list(MODELS))
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--megapixels", type=float, default=1.0)
    ap.add_argument("--quant", default="nvfp4", choices=["nvfp4", "bf16"])
    ap.add_argument("--attention", default="sage_attn")
    ap.add_argument("--score", default=None, help="only score an existing output folder")
    args = ap.parse_args()
    if args.score:
        return score(args.score)
    if not args.image:
        ap.error("--image is required to generate")
    os.makedirs(args.out, exist_ok=True)
    generate(args)
    score(args.out)


if __name__ == "__main__":
    main()
