"""Is chained-edit degradation from our speed stack? Same 3-edit chain, same seed, per config.

  python scripts/chain_check.py --tag bf16_sdpa            (one config per process)
  python scripts/chain_check.py --tag int8_sage --int8 --attention sage_attn
  python scripts/chain_check.py --sheet
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
OUT = "/workspace/mt/chain"
EDITS = ["Replace the green background with a cozy living room with a wooden bookshelf.",
         "Give the cat round golden glasses.", "Put a small red bow tie on the cat."]

def run(tag, int8, attention):
    from mtedit.session import EditSession, Engine
    extra = {"component_quantizations": {"transformer": "convrot_int8"}} if int8 else {}
    engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir=f"{OUT}/{tag}", attention=attention, **extra)
    s = EditSession(engine, "/workspace/mt/photos/972a_cat.png", crop=False, session_id=tag)
    paths = [s.path("v000")]
    for text in EDITS:
        v = s.edit(text, seed=9); s.flush(); paths.append(s.path(v.id)); print(f"[{tag}] {v.seconds:.2f}s {text[:30]}", flush=True)
    json.dump(paths, open(f"{OUT}/{tag}/paths.json", "w")); engine.shutdown()

def sheet():
    from PIL import Image, ImageDraw
    tags = [t for t in ("bf16_sdpa", "int8_sdpa", "int8_sage") if os.path.exists(f"{OUT}/{t}/paths.json")]
    rows = [[Image.open(p).convert("RGB") for p in json.load(open(f"{OUT}/{t}/paths.json"))] for t in tags]
    W = 400; th = [[im.resize((W, int(im.height * W / im.width))) for im in r] for r in rows]
    s = Image.new("RGB", (W * 4 + 30, (th[0][0].height + 20) * len(tags)), "white"); g = ImageDraw.Draw(s)
    for i, r in enumerate(th):
        for j, t in enumerate(r):
            s.paste(t, (j * (W + 10), i * (t.height + 20) + 18)); g.text((j * (W + 10) + 2, i * (t.height + 20) + 3), f"{tags[i]} turn {j}", fill=(0, 0, 0))
    s.save(f"{OUT}/sheet_chain.jpg", quality=88)
    z = Image.new("RGB", (len(tags) * 610, 600), "white")
    for i, r in enumerate(rows):
        im = r[-1]; z.paste(im.crop((im.width // 3, 0, im.width // 3 + 600, 600)), (i * 610, 0))
    z.save(f"{OUT}/zoom_chain.jpg", quality=88); print("sheet ok", tags)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag"); ap.add_argument("--int8", action="store_true")
    ap.add_argument("--attention"); ap.add_argument("--sheet", action="store_true"); a = ap.parse_args()
    sheet() if a.sheet else run(a.tag, a.int8, a.attention)
