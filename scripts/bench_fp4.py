"""GEMM microbench at Qwen-Image 2.1 DiT shapes: BF16 vs FP8 rowwise vs NVFP4 (flashinfer mm_fp4).

  python scripts/bench_fp4.py
NVFP4 timings include the per-call activation quantization (weights are quantized once).
"""
import torch, flashinfer

def timed(fn, iters=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); e.synchronize(); return s.elapsed_time(e) / iters

FP4_MAX, FP8_MAX = 6.0, 448.0
def gscale(t): return (FP8_MAX * FP4_MAX / t.float().abs().amax()).to(torch.float32)

for M in (4096, 16384):
    for N, K in ((4096, 4096), (12288, 4096), (4096, 12288)):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        flops = 2 * M * N * K
        tb = timed(lambda: x @ w.t())
        xs = (x.abs().amax(1, keepdim=True).float() / FP8_MAX); ws = (w.abs().amax(1, keepdim=True).float() / FP8_MAX)
        x8 = (x / xs).to(torch.float8_e4m3fn); w8 = (w / ws).to(torch.float8_e4m3fn)
        try: t8 = timed(lambda: torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16))
        except Exception as err: t8 = float("nan"); print("fp8 err", err)
        gw, gx = gscale(w), gscale(x)
        wq, wsf = flashinfer.fp4_quantize(w, gw)
        line = f"M={M:5d} N={N:5d} K={K:5d} | bf16 {tb:6.3f}ms {flops/tb/1e9:5.0f}TF | fp8 {t8:6.3f}ms {flops/t8/1e9:5.0f}TF"
        for backend in ("cutlass", "cudnn", "auto"):
            try:
                xq, xsf = flashinfer.fp4_quantize(x, gx)
                alpha = (1.0 / (gx * gw)).to(torch.float32)
                out = flashinfer.mm_fp4(xq, wq.t(), xsf, wsf.t(), alpha, torch.bfloat16, backend=backend)
                ref = (x @ w.t()).float(); rel = ((out.float() - ref).norm() / ref.norm()).item()
                tq = timed(lambda: flashinfer.mm_fp4(xq, wq.t(), xsf, wsf.t(), alpha, torch.bfloat16, backend=backend))
                ta = timed(lambda: flashinfer.fp4_quantize(x, gx))
                line += f" | fp4[{backend}] {tq:6.3f}ms {flops/tq/1e9:5.0f}TF +quant {ta:5.3f} relerr {rel:.3f}"
                break
            except Exception as err:
                line += f" | fp4[{backend}] fail({str(err)[:40]})"
        print(line, flush=True)
