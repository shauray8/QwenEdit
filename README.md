# mtedit: multi-turn image editing for Qwen-Image 2.1 on SGLang

An SGLang-Diffusion plugin plus a small client library. It makes repeated
edits of the same image cheap and stable, with no SGLang source changes.

## What it adds

| Feature | Where | Effect |
|---|---|---|
| Cross-request prefix K/V store | worker (`kvstore.py`, `keys.py`, `prefill.py`) | Reference images and prompt text whose K/V was computed by an earlier request are not prefilled again. Re-rolls, undo/branching and CFG branches reuse it automatically. |
| Partial prefill | worker (`prefill.py`) | When only the tail of the prefix is new (same image, new prompt), only the tail runs through the DiT. |
| VAE-latent cache | worker (`encoding.py`) | A reference image seen before is not VAE-encoded again. |
| Latent passthrough | worker (`encoding.py`, `hooks.py`) | A previous output's final latents become the next turn's condition latents, skipping the VAE decode→encode round trip. |
| Region edits | worker (`masking.py`, `hooks.py`) + session | Everything outside the boxes stays on the source trajectory and is exactly the source latents at the end; the session pastes source pixels back outside a feathered mask. |
| Sessions | client (`session.py`) | Version tree (edit, re-roll, undo, checkout), slot layouts (`chain`, `anchor`), prompt templating. |

## Why prefix reuse is exact

The Qwen-Image 2.1 DiT runs one sequence: a prefix (text tokens and reference
image blocks) and the target. The prefix uses timestep-0 modulation, so its K/V
is the same at every step, and its attention is block-causal (text causal, an
image block sees what precedes it and itself). A segment's K/V therefore
depends only on its own inputs and everything before it. `keys.py` chains a
hash over raw segment inputs (token ids, image pixels or a latent id) so equal
chain keys mean equal K/V. A request takes the longest stored prefix, prefills
only the rest (`prefill.py`, mirroring the DiT's own prefix path), and hands the
assembled K/V to the DiT's existing per-request `prefix_caches`, so the normal
cached denoising path runs unchanged.

## How it hooks into SGLang

Registered through the `sglang.multimodal_gen.plugins` entry point, so it loads
in every SGLang process (including spawned GPU workers) once installed:

- `QwenImage21EncodingStage.forward`: AROUND (replaced; a copy of upstream with marked
  additions, fingerprint-checked against upstream on first call)
- `QwenImage21DenoisingStage._predict_noise`: AROUND (store lookup, partial prefill, insert)
- `QwenImage21DenoisingStage._run_denoising_step`: AROUND (latent mask blend)
- `QwenImage21PipelineConfig.post_denoising_loop`: AROUND (keep final latents)

Per-request options travel in `diffusers_kwargs["mt"]`, which SGLang already
forwards into `batch.extra` from `DiffGenerator.generate` and from
`/v1/images/generations`:

```json
{"mt": {
  "store_output_as": "sess/v003",
  "ref_latents": {"0": "sess/v002"},
  "mask": {"boxes": [[512, 768, 640, 896]], "canvas_ref": 0, "feather_px": 16},
  "prefix_cache": true
}}
```

Requests without `mt` still get automatic prefix reuse.

## Usage

```python
from mtedit.session import Engine, EditSession

engine = Engine("Qwen/Qwen-Image-2.1-Turbo", out_dir="mt_out")
s = EditSession(engine, "board.png", layout="chain")
s.edit("Move the white pawn from e2 to e4.", boxes=[(512, 768, 640, 896), (512, 512, 640, 640)])
s.reroll()          # same edit, new seed: the whole prefix comes from the store
s.undo(); s.edit("...")
```

## Environment

| Variable | Default | |
|---|---|---|
| `MTEDIT_PREFIX_CACHE` | `1` | cross-request prefix K/V reuse |
| `MTEDIT_KV_GPU_GIB` | `24` | GPU budget for stored K/V (512 KiB per token; a 1024² reference is 4096 tokens = 2 GiB) |
| `MTEDIT_KV_CPU_GIB` | `64` | pinned-host tier; `0` drops instead of offloading |
| `MTEDIT_LATENT_SLOTS` | `512` | stored output latents / VAE encodes |
| `MTEDIT_DEBUG` | `0` | per-request reuse logging |

## Limits

- Single GPU (no TP/SP): the store holds whole-prefix K/V per device.
- A mask's canvas reference must have the target size (always true inside a session).
- Breakable CUDA graphs are untested with the store; they are slower than eager for Q21 on the PRO 6000 anyway.

## Layout

```
mtedit/         plugin + client library
tests/          CPU unit tests (pytest)
scripts/        pod helpers (rp, sync.sh, pod_setup.sh) and gpu_check.py
chess_test/     manual chess-move experiment inputs
```
