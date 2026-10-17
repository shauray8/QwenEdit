"""mm_fp4 backends at Q21 shapes, and NVFP4 attention (sm120) vs Sage2 at 2K edit shapes."""
import inspect, torch, flashinfer
from bench_attn import timed

FP4_MAX, FP8_MAX = 6.0, 448.0
gs = lambda t: (FP8_MAX * FP4_MAX / t.float().abs().amax()).to(torch.float32)
for M, N, K in ((16384, 4096, 4096), (16384, 12288, 4096), (16384, 4096, 12288), (4096, 4096, 4096)):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * .02
    gx, gw = gs(x), gs(w); xq, xsf = flashinfer.fp4_quantize(x, gx); wq, wsf = flashinfer.fp4_quantize(w, gw); a = 1 / (gx * gw)
    row = f"M={M} N={N} K={K}:"
    for b in ("cutlass", "cudnn", "cute-dsl", "b12x", "trtllm"):
        try:
            f = lambda: flashinfer.mm_fp4(xq, wq.t(), xsf, wsf.t(), a, torch.bfloat16, backend=b)
            f(); row += f" {b} {timed(f):.3f}ms |"
        except Exception as e:
            row += f" {b} x({type(e).__name__}) |"
    print(row, flush=True)

import importlib
m = importlib.import_module("flashinfer.nvfp4_attention_sm120")
print("nvfp4 attn api:", [n for n in dir(m) if not n.startswith("_")])
fwd = getattr(flashinfer, "nvfp4_attention_sm120_fwd", None)
print("fwd sig:", inspect.signature(fwd) if fwd else None)
