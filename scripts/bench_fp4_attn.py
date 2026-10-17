"""NVFP4 attention (flashinfer, sm120) vs Sage2 at Q21 edit shapes: time and error vs exact."""
import torch, flashinfer, sageattention as sa
import torch.nn.functional as F
from bench_attn import timed

for Lq, Lp in ((4096, 4130), (16384, 9216)):
    H, D, Lk = 32, 128, Lq + Lp
    q = torch.randn(1, H, Lq, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, H, Lk, D, device="cuda", dtype=torch.bfloat16); v = torch.randn(1, H, Lk, D, device="cuda", dtype=torch.bfloat16)
    ref = F.scaled_dot_product_attention(q[:, :, :1024], k, v).float()
    err = lambda o: ((o[:, :, :1024].float() - ref).norm() / ref.norm()).item()
    ts = timed(lambda: sa.sageattn(q, k, v, tensor_layout="HND"))
    es = err(sa.sageattn(q, k, v, tensor_layout="HND"))
    line = f"Lq={Lq} Lk={Lk}: sage2 {ts:.3f}ms err {es:.4f}"
    for layout, (qq, kk, vv) in (("HND", (q, k, v)),):
        try:
            packed = flashinfer.nvfp4_attention_sm120_quantize_qkv(qq, kk, vv)
            o = flashinfer.nvfp4_attention_sm120_fwd(*packed)
            o = o[0] if isinstance(o, tuple) else o
            o = o if layout == "HND" else o.transpose(1, 2)
            tq = timed(lambda: flashinfer.nvfp4_attention_sm120_quantize_qkv(qq, kk, vv))
            tf = timed(lambda: flashinfer.nvfp4_attention_sm120_fwd(*packed))
            line += f" | nvfp4[{layout}] fwd {tf:.3f}ms +quant {tq:.3f}ms err {err(o):.4f}"
        except Exception as e:
            line += f" | nvfp4[{layout}] x({str(e)[:70]})"
    print(line, flush=True)
