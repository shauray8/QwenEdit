import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
def main():
    from PIL import Image, ImageDraw
    from mtedit.session import EditSession, Engine
    e = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="/workspace/mt/r2k", attention="sage_attn", component_quantizations={"transformer": "convrot_int8"})
    out = []
    for rs in (1.0, 0.5625, 0.25):
        s = EditSession(e, "/workspace/mt/photos/972a_cat.png", megapixels=4.0, crop=False, ref_scale=rs, session_id=f"r{rs}")
        s.edit("Make it warmer.", seed=1)
        t = time.perf_counter(); v = s.edit("Replace the green background with a cozy living room with a wooden bookshelf.", seed=9, from_version="v000")
        dt = time.perf_counter() - t; s.flush(); out.append((rs, dt, s.path(v.id)))
        print(f"[ref] scale={rs} {dt:.2f}s", flush=True)
    ims = [Image.open(p).convert("RGB") for _, _, p in out]
    z = Image.new("RGB", (3 * 610, 640), "white"); g = ImageDraw.Draw(z)
    for i, (im, (rs, dt, _)) in enumerate(zip(ims, out)):
        z.paste(im.crop((900, 250, 1500, 850)), (i * 610, 40)); g.text((i * 610 + 4, 10), f"ref_scale {rs}  {dt:.2f}s", fill=(0, 0, 0))
    z.save("/tmp/refscale.jpg", quality=90); e.shutdown()
if __name__ == "__main__":
    main()
