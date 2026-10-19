"""The DiT prefix is block-causal, so a chained hash of the raw inputs up to a block identifies its
K/V exactly. Hidden states are never hashed: their low bits change with sequence length."""

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
    return _h(b"root", repr(identity).encode())


def image_content_key(image) -> bytes:
    return _h(b"img", image.mode.encode(), repr(image.size).encode(), image.tobytes())


def latent_key(latent_id: str) -> bytes:
    return _h(b"lat", latent_id.encode())


MASK63 = (1 << 63) - 1
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _as_int(digest: bytes) -> int:
    return int.from_bytes(digest[:8], "little") & MASK63


def _splitmix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x & np.uint64(MASK63)).astype(np.int64)


def _block_ids(prev: int, content: bytes, n: int) -> np.ndarray:
    base = np.uint64(_as_int(_h(prev.to_bytes(8, "little"), b"B", content)))
    with np.errstate(over="ignore"):
        return _splitmix(base + np.arange(n, dtype=np.uint64) * _GOLDEN)


def _text_id(prev: int, token: int) -> int:
    return _as_int(_h(prev.to_bytes(8, "little"), b"t", int(token).to_bytes(8, "little", signed=True)))


def virtual_ids(
    root: bytes,
    segments: Sequence[tuple[int, int, bool]],
    text_indices: Sequence[int],
    ids: Sequence[int],
    image_keys: Sequence[bytes],
    image_shapes: Sequence[tuple[int, int]],
) -> np.ndarray:
    total = segments[-1][1] if segments else 0
    out = np.empty(total, np.int64)
    prev, image = _as_int(root), 0
    for start, end, is_image in segments:
        if is_image:
            h, w = image_shapes[image]
            if (end - start) != h * w:
                raise ValueError("image segment length does not match its latent shape")
            content = image_keys[image] + np.asarray([h, w], np.int64).tobytes()
            out[start:end] = _block_ids(prev, content, end - start)
            prev = int(out[end - 1])
            image += 1
        else:
            for position in range(start, end):
                prev = _text_id(prev, ids[text_indices[position]])
                out[position] = prev
    if image != len(image_keys):
        raise ValueError("image segments do not match the reference images")
    return out


def sequence_ids(root: bytes, token_ids: Sequence[int], image_runs: Sequence[tuple[int, int, bytes]]) -> np.ndarray:
    out = np.empty(len(token_ids), np.int64)
    prev, position = _as_int(root), 0
    for start, end, key in sorted(image_runs):
        for p in range(position, start):
            prev = _text_id(prev, token_ids[p])
            out[p] = prev
        out[start:end] = _block_ids(prev, key, end - start)
        prev = int(out[end - 1]) if end > start else prev
        position = end
    for p in range(position, len(token_ids)):
        prev = _text_id(prev, token_ids[p])
        out[p] = prev
    return out
