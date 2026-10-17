"""Box edits on fixed source images against the running editor; one column per editor config.

  python scripts/demo/box_compare.py --sources          # generate the sources once (any config)
  python scripts/demo/box_compare.py --tag turbo_lock   # run the cases on the current editor
  python scripts/demo/box_compare.py --sheet            # sources + every tag, side by side, with a diff row
"""

import argparse
import io
import json
import os
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

API = "http://127.0.0.1:8000"
OUT = "/workspace/mt/boxcmp"
CASES = [
    ("loft", "Wide photograph of an empty industrial loft, raw concrete walls, polished concrete floor, tall black steel windows on the left, soft overcast daylight, 35mm editorial interior photo",
     "16:9", [{"box": [0.30, 0.52, 0.74, 0.86], "label": "a cognac leather mid-century sofa"}]),
    ("mug", "A wooden kitchen table by a window in the morning, linen napkin, a plate of croissants, soft natural light, photorealistic",
     "4:3", [{"box": [0.60, 0.45, 0.80, 0.75], "label": "a white ceramic coffee mug with steam rising"}]),
    ("smoke", "A snow-capped volcanic mountain under a bright blue sky with white cumulus clouds, crisp landscape photo",
     "16:9", [{"box": [0.40, 0.04, 0.62, 0.62], "label": "a thick column of grey volcanic smoke rising from the summit"}]),
    ("hat", "Studio portrait of a young man with curly dark hair in a navy sweater, grey seamless backdrop, soft key light, 85mm",
     "3:4", [{"box": [0.25, 0.02, 0.75, 0.30], "label": "a red wool beanie on his head"}]),
    ("lamp", "A minimal Scandinavian living room, white walls, light oak floor, grey sofa, large window, overcast daylight, photorealistic",
     "16:9", [{"box": [0.76, 0.20, 0.92, 0.85], "label": "a tall brass floor lamp with a white shade, switched on"}]),
]


def call(path, body=None, data=None, headers=None):
    req = urllib.request.Request(API + path, data=data if data is not None else json.dumps(body).encode(),
                                 headers=headers or {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def fetch(sid, vid):
    return Image.open(io.BytesIO(urllib.request.urlopen(f"{API}/api/sessions/{sid}/image/{vid}", timeout=300).read())).convert("RGB")


def upload(path):
    b = "----mtcmp"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"s.png\"\r\nContent-Type: image/png\r\n\r\n").encode() \
        + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()
    return call("/api/sessions?resolution=2k", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})


def sources():
    os.makedirs(f"{OUT}/src", exist_ok=True)
    for name, prompt, aspect, _ in CASES:
        s = call("/api/generate", {"prompt": prompt, "aspect": aspect, "resolution": "2k", "seed": 11})
        fetch(s["id"], s["head"]).save(f"{OUT}/src/{name}.png")
        print("source", name, flush=True)


def run(tag):
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    rows = {}
    for name, _, _, regions in CASES:
        s = upload(f"{OUT}/src/{name}.png")
        W, H = s["size"]
        regs = [{"box": [round(r["box"][0] * W), round(r["box"][1] * H), round(r["box"][2] * W), round(r["box"][3] * H)],
                 "label": r["label"]} for r in regions]
        t = time.perf_counter()
        r = call(f"/api/sessions/{s['id']}/edit", {"regions": regs, "instruction": "", "seed": 500, "context": False})
        dt = time.perf_counter() - t
        fetch(s["id"], r["head"]).save(f"{OUT}/{tag}/{name}.png")
        rows[name] = dict(seconds=r["versions"][r["head"]]["seconds"], wall=dt, boxes=[g["box"] for g in regs])
        print(f"[{tag}] {name} {rows[name]['seconds']:.2f}s", flush=True)
    json.dump(rows, open(f"{OUT}/{tag}/rows.json", "w"))


def sheet(width=560):
    tags = sorted(t for t in os.listdir(OUT) if os.path.exists(f"{OUT}/{t}/rows.json"))
    for name, _, _, _ in CASES:
        src = Image.open(f"{OUT}/src/{name}.png").convert("RGB")
        cols = [("source", src, None)]
        for t in tags:
            p = f"{OUT}/{t}/{name}.png"
            if os.path.exists(p):
                cols.append((f"{t} {json.load(open(f'{OUT}/{t}/rows.json'))[name]['seconds']:.1f}s", Image.open(p).convert("RGB"), p))
        k = width / src.width
        h = round(src.height * k)
        canvas = Image.new("RGB", (len(cols) * (width + 6), 2 * (h + 24)), "white")
        d = ImageDraw.Draw(canvas)
        for i, (label, im, _) in enumerate(cols):
            x = i * (width + 6)
            canvas.paste(im.resize((width, h)), (x, 0)); d.text((x + 4, h + 5), label, fill="black")
            if i:  # amplified difference to the source: anything outside the new object is drift
                diff = np.abs(np.asarray(im, np.float32) - np.asarray(src, np.float32)).mean(-1)
                dm = Image.fromarray(np.clip(diff * 6, 0, 255).astype(np.uint8)).resize((width, h))
                canvas.paste(dm.convert("RGB"), (x, h + 24))
        canvas.save(f"{OUT}/sheet_{name}.jpg", quality=88)
    print("sheets", tags)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", action="store_true"); ap.add_argument("--tag"); ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    sources() if a.sources else sheet() if a.sheet else run(a.tag)
