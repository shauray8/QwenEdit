"""Run a showcase chain from scenarios.json against the editor on the pod (localhost:8000).

  python scripts/demo/run.py tide_room [--out /workspace/mt/demo] [--speed quality]

Saves every frame (00_start.png, 01_water.png, ...), a contact sheet, and steps.json with the exact
prompt, box and timing of each step. Script-generated sessions show up in the editor's history too.
"""

import argparse
import base64
import io
import json
import os
import time
import urllib.request

from PIL import Image, ImageDraw

API = "http://127.0.0.1:8000"
HERE = os.path.dirname(os.path.abspath(__file__))


def call(path, body=None):
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = r.read()
    return data if "/image/" in path else json.loads(data)


def fetch(sid, vid):
    return Image.open(io.BytesIO(call(f"/api/sessions/{sid}/image/{vid}"))).convert("RGB")


def data_url(image):
    buf = io.BytesIO(); image.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sheet(frames, path, width=520):
    thumbs = []
    for name, im in frames:
        t = im.copy(); t.thumbnail((width, width))
        canvas = Image.new("RGB", (t.width, t.height + 28), "white")
        canvas.paste(t, (0, 0)); ImageDraw.Draw(canvas).text((6, t.height + 7), name, fill="black")
        thumbs.append(canvas)
    cols = min(4, len(thumbs)); rows = -(-len(thumbs) // cols)
    cw, ch = max(t.width for t in thumbs), max(t.height for t in thumbs)
    out = Image.new("RGB", (cols * (cw + 8), rows * (ch + 8)), "white")
    for i, t in enumerate(thumbs):
        out.paste(t, ((i % cols) * (cw + 8), (i // cols) * (ch + 8)))
    out.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--out", default="/workspace/mt/demo")
    ap.add_argument("--speed", default="quality")
    ap.add_argument("--resolution", default="2k")
    args = ap.parse_args()
    sc = json.load(open(os.path.join(HERE, "scenarios.json")))[args.scenario]
    out = os.path.join(args.out, args.scenario)
    os.makedirs(out, exist_ok=True)
    common = {"speed": args.speed, "context": False}

    t0 = time.perf_counter()
    s = call("/api/generate", {"prompt": sc["start"], "aspect": sc.get("aspect", "1:1"),
                               "resolution": args.resolution, "seed": sc.get("seed")})
    sid, (W, H) = s["id"], s["size"]
    versions = {"start": s["head"]}
    frames = [("00 start", fetch(sid, s["head"]))]
    frames[0][1].save(f"{out}/00_start.png")
    log = [{"step": "start", "prompt": sc["start"], "seconds": round(time.perf_counter() - t0, 2)}]
    print(f"session {sid} {W}x{H}")

    for k, step in enumerate(sc["steps"], 1):
        body = dict(common, seed=sc.get("seed", 0) + k)
        if step.get("from"):
            body["from_version"] = versions[step["from"]]
        regions = []
        for r in step.get("regions", []):
            x0, y0, x1, y1 = r["box"]
            reg = {"box": [round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)], "label": r.get("label", "")}
            ref = r.get("ref")
            if ref and ref.startswith("gen:"):
                g = call("/api/generate", {"prompt": ref[4:] + ", isolated on a pure white seamless background, soft even studio light",
                                           "aspect": "1:1", "resolution": "1k", "seed": k})
                item = fetch(g["id"], g["head"]); item.save(f"{out}/{k:02d}_{step['name']}_item.png")
                reg["ref"] = data_url(item)
            elif ref and ref.startswith("@"):
                reg["ref"] = data_url(Image.open(f"{out}/{ref[1:]}.png"))
            regions.append(reg)
        body["regions"] = regions
        body["instruction"] = step.get("edit", "")
        t = time.perf_counter()
        r = call(f"/api/sessions/{sid}/edit", body)
        wall = time.perf_counter() - t
        v = r["versions"][r["head"]]
        versions[step["name"]] = r["head"]
        im = fetch(sid, r["head"])
        im.save(f"{out}/{k:02d}_{step['name']}.png")
        frames.append((f"{k:02d} {step['name']}  {v['seconds']:.1f}s", im))
        log.append({"step": step["name"], "version": r["head"], "from": step.get("from"), "seconds": round(v["seconds"], 2),
                    "wall": round(wall, 2), "crop": v["timings"].get("crop"), "prompt": v["prompt"],
                    "boxes": [g["box"] for g in regions]})
        print(f"{k:02d} {step['name']:12s} {v['seconds']:5.2f}s  crop={bool(v['timings'].get('crop'))}")
    json.dump({"title": sc["title"], "session": sid, "size": [W, H], "steps": log}, open(f"{out}/steps.json", "w"), indent=1)
    sheet(frames, f"{out}/sheet.jpg")
    print("saved", out)


if __name__ == "__main__":
    main()
