"""End-to-end check of item references on boxes, against a running editor (localhost:8000).

  python scripts/demo/object_check.py [--out /workspace/mt/demo/objcheck]

Generates an empty loft and a product shot of a sofa, then places the sofa through a box with the
product image attached; saves the room, the item, the result and the exact prompt.
"""

import argparse
import base64
import io
import json
import os
import time
import urllib.request

from PIL import Image

API = "http://127.0.0.1:8000"


def call(path, body=None):
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read() if path.endswith(("png", "jpg")) or "/image/" in path else json.load(r)


def image(sid, vid):
    return Image.open(io.BytesIO(call(f"/api/sessions/{sid}/image/{vid}")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/mt/demo/objcheck")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    room = call("/api/generate", {"prompt": (
        "Wide photograph of an empty industrial loft: raw concrete walls with faint water stains, polished concrete floor, "
        "tall black steel-framed windows on the left letting in soft overcast daylight, nothing in the room, "
        "35mm lens at eye level, editorial interior photography"), "aspect": "16:9", "resolution": "2k", "seed": 7})
    item = call("/api/generate", {"prompt": (
        "Studio product photo of a mid-century cognac leather three-seat sofa with tufted cushions and thin walnut legs, "
        "three-quarter view, isolated on a pure white seamless background, soft even light"), "aspect": "4:3",
        "resolution": "1k", "seed": 3})
    sofa = image(item["id"], item["head"])
    sofa.save(f"{args.out}/item.png")
    sid = room["id"]
    W, H = room["size"]
    image(sid, room["head"]).save(f"{args.out}/room.png")
    buf = io.BytesIO(); sofa.convert("RGB").save(buf, "JPEG", quality=92)
    ref = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    box = [round(W * 0.30), round(H * 0.52), round(W * 0.74), round(H * 0.86)]
    t0 = time.perf_counter()
    out = call(f"/api/sessions/{sid}/edit", {"regions": [{"box": box, "label": "place this sofa on the floor", "ref": ref}],
                                              "speed": "quality", "seed": 11})
    wall = time.perf_counter() - t0
    v = out["versions"][out["head"]]
    image(sid, out["head"]).save(f"{args.out}/placed.png")
    print(json.dumps({"seconds": v["seconds"], "wall": round(wall, 2), "timings": v["timings"], "prompt": v["prompt"], "box": box}, indent=1))


if __name__ == "__main__":
    main()
