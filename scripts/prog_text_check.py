import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
def main():
    from PIL import Image, ImageDraw
    from mtedit.session import EditSession, Engine
    e = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="/workspace/mt/prog", attention="sage_attn", component_quantizations={"transformer": "convrot_int8"})
    res = []
    for k in (0, 3, 4):
        s = EditSession(e, "/workspace/mt/photos/a6d2_TI2I_Qwen_Image_Edit_Input.jpg", megapixels=4.0, crop=False, ref_scale=0.5625, session_id=f"k{k}")
        s.progressive = k
        s.edit("Make it warmer.", seed=1)
        t = time.perf_counter(); v = s.edit("Change the large orange text \"SGL\" to read \"EDIT FAST\" in the same style.", seed=9, from_version="v000")
        dt = time.perf_counter() - t; s.flush(); res.append((k, dt, s.path(v.id)))
        print(f"[prog] k={k} {dt:.2f}s", flush=True)
    ims = [Image.open(p).convert("RGB") for *_, p in res]
    full = Image.new("RGB", (3 * 500, 380), "white"); crop = Image.new("RGB", (3 * 500, 520), "white"); g1, g2 = ImageDraw.Draw(full), ImageDraw.Draw(crop)
    for i, (im, (k, dt, _)) in enumerate(zip(ims, res)):
        t = im.copy(); t.thumbnail((490, 360)); full.paste(t, (i * 500, 20)); g1.text((i * 500 + 3, 4), f"k={k} {dt:.2f}s", fill=(0, 0, 0))
        crop.paste(im.crop((1300, 100, 2400, 600)).resize((490, 223)), (i * 500, 20)); g2.text((i * 500 + 3, 4), f"k={k}", fill=(0, 0, 0))
    full.save("/tmp/progtxt_full.jpg", quality=88); crop.save("/tmp/progtxt_crop.jpg", quality=88); e.shutdown()
if __name__ == "__main__":
    main()
