# Best multi-turn editing: design and next experiments

Constraint for everything below: **no inference slowdown.** Anything that adds work must
either run off the critical path (while the user thinks or types) or pay for itself.

## 1. What "multi-turn" has to mean

The image model is single-turn: one image, one instruction. Its prefix K/V cache is
attention state for *that* request, not memory of earlier instructions, so "make it red"
after "add a sun" is ambiguous to it. Multi-turn behavior has to be built around it:

| Layer | What it gives | Status |
|---|---|---|
| Version tree, re-roll, undo, branch | navigation | done |
| Latent passthrough, locked outside | no drift across turns | done |
| Text context (resolver) | "it" → "the sun" | done (needs GPU test) |
| **Scene memory (objects)** | *where* "it" is, precisely | next |
| **Auto change-masks** | protect what an unboxed edit didn't mean to touch | next |
| Identity anchor | long chains keep faces/products | exists (`anchor` layout) |

### 1a. Scene memory: an object registry
Every edit that adds or changes something records an entry:
`{name, version, box, mask}`. Names come from the resolver ("sun", "red mug").
The mask is **measured, not guessed**: after an edit, `|output − parent|` inside the box,
thresholded and cleaned up (morphological close, largest component). It costs about 1 ms and needs no segmentation model.

Then "make it red" resolves to an object, and the edit uses **that object's mask**
(tight), not the original box (loose). "Move it left" shifts the stored box. "Remove the
mug" knows exactly which pixels. The resolver gets the registry (`[1] sun (top-left)`)
instead of raw history lines, which also makes its job easier.

### 1b. Auto change-masks for unboxed edits
Today an unboxed "make the sun red" regenerates the whole frame, and everything drifts a
little. Instead: run the edit, compute the change mask (output vs. parent, in latent space,
where it's cheap), dilate it, and paste the parent back everywhere else. Global edits
(lighting, style) change everything and are left alone: the resolver's `mode=global`
switches this off. Cost: about 1 ms. It's a large win, because most real edits are local
even when no box is drawn.

### 1c. Mode-aware region edits (fixes the half-person drift)
| Mode | Start | Mask | Marks |
|---|---|---|---|
| add | pure noise (σ=1) | box | outlined |
| modify | source noised to σ≈0.85 (SDEdit, 3 steps) | object mask if known, else box | outlined |
| remove | pure noise | box, grown slightly | outlined |
| global | pure noise | none (or change-mask off) | none |

Modify is now SDEdit, which keeps geometry and is faster. The masks feather inward too.
Still to do: **cut detection.** If the change mask touches the box edge along a long stretch,
the box cut through an object (the half-person case). Grow the box to the object's
change-mask extent and re-run, or warn in the UI. The re-run only happens when needed.

### 1d. Hiding the resolver's latency
- **Skip it when not needed**: no pronoun, demonstrative or comparative ("it", "that", "them",
  "bigger", "again"), and no reference words, means no call. Most first edits pay nothing.
- **Speculative**: the editor calls `/resolve` debounced while you type (~300 ms after the
  last key). By the time you press Enter the answer is usually there, so the cost at submit is ~0.
- Small model on the same GPU (Qwen3-4B, about 10 GB, about 0.15–0.3 s for 40 tokens).

### 1e. Things to drop or demote
- The PE-I2I enhancer (3–70 s) leaves the main flow. It's at most an explicit tool, and the resolver
  covers the common need: clear, self-contained instructions.

## 2. Speed: what's left and how to measure it

Measured edit step (INT8 ConvRot + fusions, 1024², 4.1k target + 4.1k prefix): ~150 ms
GPU, with INT8 GEMMs 52%, attention 37% before Sage2, and rotate/LN/residual about 7%.
**The same profile showed about 143 ms of CPU time per step**: the step is close to
launch-bound. That changes the priority list.

### Ranked experiments
1. **CUDA graphs for the denoise step** (breakable CUDA graph already exists in SGLang). The earlier
   test was BF16 t2i at 40 steps, where GPU time dominated. With INT8 and many small fused
   kernels, CPU launch time ≈ GPU time, so graphs should cut real latency. The prefix-cache path
   already runs with populated caches from step 0 (mtedit fills them before the first call), which is
   the condition graph replay needs.
2. **Cached quantized prefix K/V for Sage2**: prefix K/V is fixed for the whole request, but
   Sage re-quantizes it every step in every layer. Quantize once (INT8 K, FP8 V, with scales), keep
   it, and split attention into prefix-part and target-part, merged by log-sum-exp. That saves the
   quantize pass over 4.1k tokens × 32 layers × 8 steps, and opens the next item.
3. **Prefix K/V in FP8 in the pools**: half the memory, half the bandwidth for assembly, and it feeds
   item 2 directly.
4. **QK-norm + RoPE into the QKV GEMM epilogue** (~4% of the step), and **residual-gate-add
   into the to_out/down GEMM epilogues** (~1%).
5. **NVFP4 GEMMs** (Blackwell): up to 2× on the 52% GEMM share, behind a strict quality gate.
6. **Last-step caching** (First-Block-Cache style) for the final 1–2 Turbo steps: only behind the gate.
7. **Decode**: VAE decode of the crop window only (crop edits already do this); tile-parallel full decode.

### The gate (every change)
- Kernel microbench at Q21 edit shapes, before vs. after.
- End-to-end: `bench_sessions.py` in fresh processes (speed must not regress).
- Quality: same-seed PSNR vs. the current default on the photo suite, plus a visual sheet.
  Bit-exact changes (fusions) must stay byte-identical, as your ConvRot fusions do.

## 3. Order of work on the next GPU session
1. Bring the pod up (`make setup`), run the resolver tests: "add a sun" → "make it red".
2. Scene memory + change-masks + cut detection (quality, ~0 ms).
3. Speculative resolver from the editor (latency hiding).
4. CUDA graphs on the INT8 edit step (largest speed lever left).
5. Cached quantized prefix K/V for Sage2, FP8 pools.
6. Epilogue fusions, then NVFP4 behind the gate.
