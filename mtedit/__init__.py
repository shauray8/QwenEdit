"""Multi-turn image editing layer for Qwen-Image 2.1 on SGLang-Diffusion.

Worker side (loaded as an SGLang plugin, see ``plugin.py``):
  * a cross-request prefix K/V store keyed by content (``kvstore.py``, ``keys.py``),
    filled and reused around the DiT's own per-request prefix cache (``prefill.py``);
  * latent passthrough: a previous output's final latents become a reference's
    condition latents, skipping the VAE round trip;
  * region edits: a latent mask that keeps everything outside the boxes on the
    source trajectory (``masking.py``).

Client side (``session.py``): an edit session with a version tree over
``DiffGenerator`` that builds the reference layout and request payloads.
"""

__version__ = "0.1.0"
