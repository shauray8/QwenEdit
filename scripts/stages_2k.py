import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
def main():
    from mtedit.session import EditSession, Engine
    e = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="/workspace/mt/s2k", attention="sage_attn", component_quantizations={"transformer": "convrot_int8"})
    s = EditSession(e, "/workspace/mt/photos/972a_cat.png", megapixels=4.0, crop=False, session_id="s")
    s.edit("Make it warmer.", seed=1)
    print("=== MEASURED ===", flush=True)
    t = time.perf_counter(); v = s.edit("Replace the green background with a cozy living room.", seed=9, from_version="v000")
    print(f"TOTAL {time.perf_counter() - t:.2f}s timings={v.timings}", flush=True)
    s.flush(); e.shutdown()
if __name__ == "__main__":
    main()
