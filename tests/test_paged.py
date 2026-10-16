"""PagedKVPool over SGLang's RadixCache + TokenToKVPoolAllocator (needs sglang; runs on the pod)."""

import numpy as np
import pytest
import torch

pytest.importorskip("sglang.srt.mem_cache.radix_cache")
from mtedit.paged import PagedKVPool  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make(tokens):
    per_token = 2 * 2 * 1 * 4 * 2  # layers=2, K+V, heads=1, dim=4, bf16
    return PagedKVPool("t", 2, 1, 4, tokens * per_token, torch.bfloat16, DEVICE)


def fill(pool, vids, value):
    match = pool.match(vids)
    n = len(vids) - match.length
    slots = pool.alloc(n)
    pool.write(slots, torch.full((2, 2, n, 1, 4), value, dtype=torch.bfloat16, device=DEVICE))
    idx = torch.cat([match.indices, slots]) if match.length else slots
    pool.insert(vids, idx, owned_from=match.length)
    match.release()


def test_match_extends_and_reads_back():
    pool = make(64)
    a = np.arange(1, 9, dtype=np.int64)
    fill(pool, a, 1.0)
    b = np.concatenate([a[:5], [100, 101]])
    m = pool.match(b)
    assert m.length == 5
    assert torch.all(pool.gather(m.indices) == 1.0)
    m.release()
    fill(pool, b, 2.0)
    m = pool.match(b)
    assert m.length == 7
    got = pool.gather(m.indices)
    assert torch.all(got[:, :, :5] == 1.0) and torch.all(got[:, :, 5:] == 2.0)
    m.release()


def test_eviction_respects_locks():
    pool = make(16)
    a = np.arange(1, 11, dtype=np.int64)
    fill(pool, a, 1.0)
    held = pool.match(a)  # locked
    assert pool.alloc(10) is None or pool.match(a).length == 10  # cannot evict the locked prefix
    held.release()
    b = np.arange(1000, 1012, dtype=np.int64)
    fill(pool, b, 3.0)  # now evicts a
    assert pool.match(a).length < 10
    assert pool.usage()["evicted_tokens"] >= 6


def test_contiguous_run_is_a_view():
    pool = make(64)
    a = np.arange(1, 9, dtype=np.int64)
    fill(pool, a, 1.0)
    m = pool.match(a)
    view = pool.gather(m.indices)
    assert view.data_ptr() == pool.kv[:, :, int(m.indices[0])].data_ptr()
    m.release()
