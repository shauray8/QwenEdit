# Multi Turn Edit with Qwen Image 2.1
Fast(ish) multi-turn edit for Qwen-Image 2.1. Tested on a 6k pro.

https://github.com/user-attachments/assets/f5ded4fb-e019-4cbf-9fdf-a6d46ba684d9

Edit latency on an RTX PRO 6000 (NVFP4, 8 steps): 1K full image 0.9 s, 2K full image 4.4 s, 2K box edit 3.9–5.2 s, small box 0.4–1 s.


## Install

Needs an NVIDIA GPU and a CUDA toolkit (`CUDA_HOME`). NVFP4 needs Blackwell; use `--quant int8` otherwise.

```
python -m venv .venv && source .venv/bin/activate
make setup
```

## Run

```
python -m multi_turn.app.server            # http://127.0.0.1:8000
python -m multi_turn.app.server --share    # plus a public *.gradio.live link
```

Optional helpers, each on its own:

```
make resolver   # follow-up rewriting   -> --resolver-url http://127.0.0.1:30001
make enhancer   # prompt enhancer       -> --enhancer-url http://127.0.0.1:30000
```

## Flags

| Flag | Default | |
|---|---|---|
| `--model` | `Qwen/Qwen-Image-2.1-Turbo` | model path or HF id |
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

Running the resolver or enhancer on the same GPU: lower the caches, e.g. `MULTI_TURN_KV_GPU_GIB=16 MULTI_TURN_ENC_GPU_GIB=3`.

## Test

```
make test                                  # unit tests
python scripts/gpu_check.py                # cache correctness on the GPU
```
