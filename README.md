# Multi Turn Edit with Qwen Image 2.1
Fast(ish) multi-turn edit for Qwen-Image 2.1. Tested on a 6k pro.



https://github.com/user-attachments/assets/bb136139-1d88-4300-8d26-4bbef5d2a13c


Edit latency on an RTX PRO 6000 (NVFP4, 8 steps): 1K full image 0.9 s, 2K full image 4.4 s, 2K box edit 3.9–5.2 s, small box 0.4–1 s.
Fibo 2.0 Edit Turbo (NVFP4, 4 steps, always at its trained ~1 MP): full image and box edit 1.3 s at 1K, 1.5–1.6 s at 2K
(RTX 5090).


## Install

Needs an NVIDIA GPU (PRO 6000 tested) and a CUDA toolkit (`CUDA_HOME`). NVFP4 needs Blackwell; use `--quant int8` otherwise.

```
python -m venv .venv && source .venv/bin/activate
make setup
```

## Run

```
python -m multi_turn.app.server            # http://127.0.0.1:8000
python -m multi_turn.app.server --share    # plus a public *.gradio.live link
python -m multi_turn.app.server --models qwen,fibo   # both models, picked per edit in the editor
```

Fibo 2.0 Edit Turbo (`briaai/Fibo-2.0-edit-turbo`) is gated on Hugging Face (`hf auth login` first) and CC BY-NC.
Both models on one 96 GB GPU need smaller Qwen caches:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MULTI_TURN_KV_GPU_GIB=6 MULTI_TURN_ENC_GPU_GIB=2`.
Fibo follows Bria's edit setup: 4 steps, no CFG, the trained ~1 MP size for the aspect ratio (2K sessions resize its
result up; 2K box edits crop a ~1 MP window), boxes greyed out and named "the masked region".
Bria's VGL prompt step is opt-in: `pip install bria-vgl` and `--fibo-llm openai/<vlm>` with `OPENAI_API_BASE`.

Optional helpers, each on its own:

```
make resolver   # follow-up rewriting   -> --resolver-url http://127.0.0.1:30001
make enhancer   # prompt enhancer       -> --enhancer-url http://127.0.0.1:30000
```

## Flags

| Flag | Default | |
|---|---|---|
| `--model` | `Qwen/Qwen-Image-2.1-Turbo` | Qwen model path or HF id |
| `--models` | `qwen` | comma-separated `qwen`, `fibo`; the first is the default |
| `--fibo-llm` | | litellm model for Bria's VGL prompt step (opt-in) |
| `--quant` | `nvfp4` | `nvfp4`, `int8` or `bf16` DiT |
| `--int8` | | same as `--quant int8` |
| `--host` / `--port` | `127.0.0.1` / `8000` | |
| `--share` | off | public Gradio tunnel |
| `--out` | `mt_out` | output folder |
| `--resolver-url` | | SGLang server for follow-ups |
| `--enhancer-url` | | SGLang server running Qwen-Image-2.1-PE-I2I |
| `--no-graphs` | off | disable CUDA graphs |
| `--steps` | model's | denoising steps (base model) |
| `--sigmas` | Turbo's 8 | schedule for the Quality preset, e.g. `1,0.75,0.5,0.25` |
| `--guidance` | | CFG scale (base model) |
| `--lora` | | LoRA merged into the weights at load |

## Environment

| Variable | Default | |
|---|---|---|
| `MULTI_TURN_KV_GPU_GIB` | `24` | DiT prefix cache size |
| `MULTI_TURN_ENC_GPU_GIB` | `4` | text-encoder prefix cache size |
| `MULTI_TURN_PREFIX_CACHE` | `1` | DiT prefix reuse |
| `MULTI_TURN_ENCODER_CACHE` | `1` | text-encoder prefix reuse |
| `MULTI_TURN_LOCK` | `0.75` | box-edit change lock sigma, `0` disables |
| `MULTI_TURN_DEBUG` | `0` | log cache reuse per request |
| `MULTI_TURN_FIBO_COMPILE` | `1` | `torch.compile` Fibo's norms, modulation and RoPE, `0` keeps them eager |

Running the resolver or enhancer on the same GPU: lower the caches, e.g. `MULTI_TURN_KV_GPU_GIB=16 MULTI_TURN_ENC_GPU_GIB=3`.

## Test

```
make test                                  # unit tests
python scripts/gpu_check.py                # cache correctness on the GPU
```
