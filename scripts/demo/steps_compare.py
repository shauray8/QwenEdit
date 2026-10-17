"""Same edits at several speed presets (same seed) on the running editor; one sheet per edit.

  python scripts/demo/steps_compare.py [--speeds quality,turbo12] [--out /workspace/mt/demo/steps]
"""

import argparse
import io
import json
import os
import urllib.request

from PIL import Image, ImageDraw

API = "http://127.0.0.1:8000"
CASES = [
    ("portrait", "Close-up portrait of an elderly fisherman with a weathered face and a knitted wool cap, soft overcast light, "
     "harbor in the background, 85mm photo", "Make it a golden-hour sunset with warm rim light on his face", None),
    ("loft", "Wide photograph of an empty industrial loft, concrete walls, tall steel windows, soft daylight, 35mm",
     "", [{"box": [0.30, 0.52, 0.74, 0.86], "label": "a low curved cream boucle sofa"}]),
    ("smoke", "A snow-capped volcanic mountain under a bright blue sky with big white cumulus clouds, crisp landscape photo",
     "", [{"box": [0.40, 0.04, 0.62, 0.62], "label": "a thick column of grey volcanic smoke rising from the summit"}]),
    ("street", "Rainy neon-lit Tokyo alley at night, wet asphalt reflections, small shop signs, cinematic 35mm",
     "Turn it into a snowy winter night, snow on every surface and falling flakes, keep the neon", None),
]


def call(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def fetch(sid, vid):
    return Image.open(io.BytesIO(urllib.request.urlopen(f"{API}/api/sessions/{sid}/image/{vid}", timeout=300).read())).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", default="quality,turbo12", help="presets, or steps:N for the model's schedule at N steps")
    ap.add_argument("--out", default="/workspace/mt/demo/steps")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    speeds = args.speeds.split(",")
    for name, start, edit, regions in CASES:
        g = call("/api/generate", {"prompt": start, "aspect": "16:9", "resolution": "2k", "seed": 4})
        W, H = g["size"]
        base = fetch(g["id"], g["head"])
        tiles = [("source", base)]
        for sp in speeds:
            reg = [{"box": [round(r["box"][0] * W), round(r["box"][1] * H), round(r["box"][2] * W), round(r["box"][3] * H)],
                    "label": r["label"]} for r in (regions or [])]
            body = {"instruction": edit, "regions": reg, "seed": 9, "from_version": g["head"], "context": False}
            body.update({"speed": "quality", "steps": int(sp[6:])} if sp.startswith("steps:") else {"speed": sp})
            r = call(f"/api/sessions/{g['id']}/edit", body)
            v = r["versions"][r["head"]]
            im = fetch(g["id"], r["head"]); im.save(f"{args.out}/{name}_{sp.replace(':', '')}.png")
            tiles.append((f"{sp}  {v['seconds']:.2f}s", im))
            print(name, sp, round(v["seconds"], 2))
        tw = 900
        th = round(tw * H / W)
        sheet = Image.new("RGB", (len(tiles) * (tw + 8), th + 30), "white")
        for i, (label, im) in enumerate(tiles):
            sheet.paste(im.resize((tw, th)), (i * (tw + 8), 0))
            ImageDraw.Draw(sheet).text((i * (tw + 8) + 6, th + 8), label, fill="black")
        sheet.save(f"{args.out}/{name}.jpg", quality=90)


if __name__ == "__main__":
    main()
