"""Try a DiT quantization method: load, time a fixed 1K and 2K edit, save outputs. One method per process.

  python scripts/quant_try.py convrot_int8|nvfp4|mxfp4|mxfp8
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    from mtedit.session import EditSession, Engine
    method = sys.argv[1]
    e = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir=f"/workspace/mt/q/{method}", attention="sage_attn",
               component_quantizations={"transformer": method})
    for mp in (1.0, 4.0):
        s = EditSession(e, "/workspace/mt/photos/972a_cat.png", megapixels=mp, crop=False, session_id=f"m{mp}")
        s.edit("Make it warmer.", seed=1)
        t = time.perf_counter(); v = s.edit("Replace the green background with a cozy living room with a wooden bookshelf.", seed=9, from_version="v000")
        dt = time.perf_counter() - t; s.flush()
        print(f"[quant] {method} {mp:.0f}MP {dt:.2f}s {s.path(v.id)}", flush=True)
    e.shutdown()

if __name__ == "__main__":
    main()
