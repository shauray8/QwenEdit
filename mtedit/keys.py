"""Content keys for prefix segments.

The DiT prefix is block-causal (text tokens causal, each image block attends to
everything before it and itself), so the K/V of a segment depends only on that
segment's inputs, its RoPE positions and everything before it. A chained hash
over the segment inputs therefore identifies the K/V exactly: two requests that
share the first n chain keys share the first n segments' K/V.

Keys are built from raw inputs (token ids, image bytes or a latent id), never
from encoder hidden states: re-running the text encoder on a longer sequence can
change low bits of the earlier rows, which would make float hashes miss.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

DIGEST = 16


def _h(*parts: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=DIGEST)
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.digest()


def root_key(*identity: object) -> bytes:
    """Everything that changes the weights or the K/V math: model, quantization, LoRA epoch."""
    return _h(b"root", repr(identity).encode())


def image_content_key(image) -> bytes:
    """Key of a resized PIL reference image as it enters the VAE (mode, size and pixels)."""
    return _h(b"img", image.mode.encode(), repr(image.size).encode(), image.tobytes())


def latent_key(latent_id: str) -> bytes:
    return _h(b"lat", latent_id.encode())


def segment_chain(
    root: bytes,
    segments: Sequence[tuple[int, int, bool]],
    text_indices: Sequence[int],
    ids: Sequence[int],
    image_keys: Sequence[bytes],
    image_shapes: Sequence[tuple[int, int]],
) -> list[tuple[int, int, bytes]]:
    """Chained keys for ``layout["segments"]``.

    ``text_indices[p]`` maps prefix position ``p`` to its row in the collapsed
    encoder sequence, whose token ids are ``ids``; image segments consume
    ``image_keys`` in order. ``image_shapes`` are the latent (h, w) per image.
    """
    chain, key, image = [], root, 0
    for start, end, is_image in segments:
        if is_image:
            h, w = image_shapes[image]
            if (end - start) != h * w:
                raise ValueError("image segment length does not match its latent shape")
            key = _h(key, b"I", image_keys[image], np.asarray([h, w], np.int64).tobytes())
            image += 1
        else:
            rows = np.asarray([ids[i] for i in text_indices[start:end]], np.int64)
            key = _h(key, b"T", rows.tobytes())
        chain.append((start, end, key))
    if image != len(image_keys):
        raise ValueError("image segments do not match the reference images")
    return chain
