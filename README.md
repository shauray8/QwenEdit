# mtedit: multi-turn image editing for Qwen-Image 2.1 on SGLang

An SGLang-Diffusion plugin, a session library and a small server for
multi-turn, region-aware editing: any sequence of add / remove / replace /
restyle edits on any image, with optional boxes, re-roll, undo and branching.
SGLang's source is not modified; everything hooks in through its plugin system.

## What it does

| Feature | Where | Effect |
|---|---|---|
| DiT prefix K/V reuse across requests | `paged.py`, `keys.py`, `prefill.py`, `hooks.py` | Reference images and prompt text already prefilled by an earlier request are not prefilled again. Re-rolls, undo/branching, re-prompting and CFG branches reuse it. A full hit hands the DiT zero-copy views into the pool. |
| Text-encoder (Qwen3-VL) prefix reuse | `encoder_cache.py` | Same for the VLM: a full hit skips the encoder; a partial hit runs only the new tail (bottom-right causal mask, per-image vision features, deepstack). Cold requests run upstream unchanged. |
| Paged pools on SGLang's LLM cache | `paged.py` | `sglang.srt` `RadixCache` (token radix tree, node splitting, lock refs, leaf-first LRU eviction) indexes a preallocated pool with `TokenToKVPoolAllocator` slots. |
| Latent passthrough | `encoding.py`, `hooks.py` | A previous output's final latents become the next turn's condition latents (no VAE decode→encode drift). |
| Region edits | `encoding.py`, `masking.py`, `session.py` | Boxes are outlined on the canvas the model sees (Qwen-Image 2.1's mark-guided local editing) and named in the prompt; a latent mask keeps everything outside on the clean source; the session pastes the source pixels back. |
| INT8 ConvRot | `Engine(..., component_quantizations={"transformer": "convrot_int8"})` | SGLang's online INT8 DiT path; all cache paths verified on it. |
| Multi-region edits | `session.py` | `regions=[(box, "what goes here"), ...]`: each box gets its own outline color and instruction in one edit. |
| Text-to-image | `Engine.text_to_image` | Generated bases keep their latents for the first edit. |
| Prefix warming | `EditSession(prewarm=True)` | After an edit, the next turn's prefix is prefilled in the background. |
| Sessions, workspace, API | `session.py`, `gradio_app.py`, `server.py` | Version tree with edit / re-roll / undo / checkout, `chain` and `anchor` layouts; Gradio editing workspace; FastAPI JSON API. |

## Why prefix reuse is exact

The DiT runs one sequence: a prefix (text tokens and reference image blocks)
then the target. The prefix uses timestep-0 modulation, so its K/V is the same
at every step, and its attention is block-causal (text causal; an image block
sees what precedes it and itself). A position's K/V therefore depends only on
the inputs up to and including its block. `keys.virtual_ids` gives every prefix
position a chained 63-bit id over raw inputs (token ids, image pixels or a
latent id): text positions chain per token, and every position of an image
block includes the block's content key, so blocks match whole or not at all.
Equal ids mean equal K/V; the radix tree finds the longest stored prefix. The
text encoder is plain causal, so the same scheme applies per token.

## Correctness (RTX PRO 6000, Turbo, 1024², 8 steps)

- Cache miss and full hit are byte-identical to the uncached path, in BF16 and INT8.
- Partial DiT hits (cached image, new prompt tail) are byte-identical in INT8 and 52.5 dB in BF16.
- Edits that reuse the text encoder's prefix differ from upstream only in low bits
  (46–61 dB PSNR): a tail-only forward cannot reproduce the full-sequence kernels bit for bit.
- Photo suite (`scripts/general_check.py`): add, remove, replace, global restyle and
  text edits on five photos with re-roll and undo; cached and uncached sessions are
  visually identical, and boxed edits leave the outside unchanged (≤ 2 levels at the feather).

## How it hooks into SGLang

Registered through the `sglang.multimodal_gen.plugins` entry point, so it loads
in every SGLang process (including spawned GPU workers) once installed:

- `QwenImage21EncodingStage.forward`: AROUND, replaced by a copy of upstream with marked
  additions; fingerprint-checked against upstream on first call
- `QwenImage21DenoisingStage._predict_noise`: AROUND (pool lookup, partial prefill, insert)
- `QwenImage21DenoisingStage._run_denoising_step`: AROUND (latent mask blend)
- `QwenImage21PipelineConfig.post_denoising_loop`: AROUND (keep final latents, release pool locks)

Per-request options travel in `diffusers_kwargs["mt"]`, which SGLang already
forwards into `batch.extra` from `DiffGenerator.generate` and `/v1/images/generations`:

```json
{"mt": {
  "store_output_as": "sess/v003",
  "ref_latents": {"0": "sess/v002"},
  "mask": {"boxes": [[512, 768, 640, 896]], "canvas_ref": 0, "feather_px": 16, "annotate": true},
  "prefix_cache": true
}}
```

Requests without `mt` still get automatic prefix reuse.

## Usage

```python
from mtedit.session import Engine, EditSession

engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="mt_out",
                component_quantizations={"transformer": "convrot_int8"})
s = EditSession(engine, "photo.jpg")                      # layout="anchor" keeps the original as image 1
s.edit("Add a white ceramic mug of coffee on the table.", boxes=[(40, 600, 300, 900)])
s.reroll()                                                # same edit, new seed; prefix fully cached
s.undo()
s.edit("Replace the background with a snowy forest.")    # no box: a global edit
```

Editing workspace (Gradio; `--share` gives a public link):

```
CUDA_HOME=/usr/local/cuda-13.0 python -m mtedit.gradio_app --int8 --share
```

Generate a base image or upload one, drag boxes on the canvas and label each
with what should be there (colors are assigned automatically; each box is
outlined in its own color for the model and named in the prompt), add an
optional global instruction, apply. Re-roll, undo, and click any version to
continue or branch from it. A plain JSON API (`python -m mtedit.server`) exposes
the same session operations.

## Environment

| Variable | Default | |
|---|---|---|
| `MTEDIT_PREFIX_CACHE` | `1` | DiT prefix reuse |
| `MTEDIT_ENCODER_CACHE` | `1` | text-encoder prefix reuse |
| `MTEDIT_KV_GPU_GIB` | `24` | DiT pool (512 KiB/token; a 1024² reference is 4096 tokens = 2 GiB) |
| `MTEDIT_ENC_GPU_GIB` | `4` | encoder pool (~152 KiB/token) |
| `MTEDIT_MIN_REUSE` / `MTEDIT_ENC_MIN_REUSE` | `256` / `128` | shorter matches run the upstream fused prefill instead |
| `MTEDIT_LATENT_SLOTS` | `512` | stored output latents / VAE encodes |
| `MTEDIT_DEBUG` | `0` | per-request reuse and encoder phase timing |

## Limits

- Single GPU (no TP/SP): the pools hold whole-prefix K/V per device. No host-memory tier yet.
- A mask's canvas reference must have the target size (always true inside a session).
- Boxes are hard constraints: an object that extends past its box is cut at the box.
- Breakable CUDA graphs are untested with the pools (slower than eager for Q21 on this card).

## Layout

```
mtedit/         plugin, pools, encoder cache, sessions, server, playground
tests/          unit tests (paged-pool tests need sglang)
scripts/        pod helpers (rp, sync.sh, pod_setup.sh), gpu_check, general_check, bench_sessions
chess_test/     manual chess-move experiment inputs
```
