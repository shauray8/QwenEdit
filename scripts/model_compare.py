"""Compare few-step models on the same edits + one t2i (same seeds); one variant per process.

  python scripts/model_compare.py --variant turbo|viggle|lightning [--int8]
  python scripts/model_compare.py --sheet       # combine all variants into sheet_models.png
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from steps_check import EDITS  # noqa: E402

OUT = "/workspace/mt/models"
VARIANTS = {
    "turbo": dict(model="Qwen/Qwen-Image-2.1-Turbo", lora=None, sigmas=None),
    "viggle": dict(model="Qwen/Qwen-Image-2.1", lora="/workspace/loras/viggle/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
                   sigmas=[1.0, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6]),
    "lightning": dict(model="Qwen/Qwen-Image-2.1", lora="/workspace/loras/lightning/pytorch_lora_weights.safetensors",
                      sigmas=[1.0, 0.75, 0.5, 0.25]),
}
T2I = "A neon shop sign that reads \"QWEN EDIT\", rainy night, reflections on wet pavement, photorealistic"


def run(name, int8):
    from mtedit.session import EditSession, Engine
    v = VARIANTS[name]
    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if int8 else {}
    if v["lora"]:
        extra.update(lora_path=v["lora"], lora_merge_mode="dynamic" if int8 else "merge")
    engine = Engine(v["model"], out_dir=f"{OUT}/{name}", attention="sage_attn", **extra)
    rows = {}
    w = EditSession(engine, "/workspace/mt/photos/972a_cat.png", crop=False, sigmas=v["sigmas"], session_id="warm")
    w.edit("Make it warmer.", seed=1)
    for photo, text, frac in EDITS:
        s = EditSession(engine, os.path.join("/workspace/mt/photos", photo), crop=False, sigmas=v["sigmas"], session_id=photo[:4])
        W, H = s.size
        box = [(frac[0] * W, frac[1] * H, frac[2] * W, frac[3] * H)] if frac else None
        t = time.perf_counter(); ver = s.edit(text, boxes=box, seed=500); dt = time.perf_counter() - t
        s.flush(); rows[photo[:4]] = dict(path=s.path(ver.id), seconds=dt, box=box)
        print(f"[{name}] {photo[:4]} {dt:.2f}s", flush=True)
    t = time.perf_counter()
    img, _ = engine.text_to_image(T2I, width=1024, height=1024, seed=42) if not v["sigmas"] else (None, None)
    if v["sigmas"]:
        img, _ = engine.run(prompt=T2I, images=[], width=1024, height=1024, seed=42, mt={"sigmas": v["sigmas"]}, name="t2i")
    dt = time.perf_counter() - t
    p = f"{OUT}/{name}/t2i.png"; img.save(p); rows["t2i"] = dict(path=p, seconds=dt, box=None)
    print(f"[{name}] t2i {dt:.2f}s", flush=True)
    json.dump(rows, open(f"{OUT}/{name}/rows.json", "w"))
    engine.shutdown()


def combine():
    from general_check import sheet
    names = [n for n in VARIANTS if os.path.exists(f"{OUT}/{n}/rows.json")]
    data = {n: json.load(open(f"{OUT}/{n}/rows.json")) for n in names}
    keys = list(data[names[0]])
    rows = [[data[n][k]["path"] for n in names] for k in keys]
    boxes = [[data[n][k]["box"] for n in names] for k in keys]
    labels = [[f"{n} {data[n][k]['seconds']:.2f}s" for n in names] for k in keys]
    sheet(f"{OUT}/sheet_models.png", rows, boxes, labels)
    print("sheet ok", names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--int8", action="store_true"); ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    combine() if a.sheet else run(a.variant, a.int8)
