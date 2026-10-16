"""CPU tests for the pieces that need no GPU or SGLang: keys, the K/V store, masks."""

import torch
from PIL import Image

from mtedit import keys
from mtedit.kvstore import PrefixKVStore
from mtedit.masking import MaskState, rasterize
from mtedit.request import MaskSpec, MTRequest

# a layout like build_layout's: text [0,3), image 2x2 [3,7), text [7,9)
SEGMENTS = ((0, 3, False), (3, 7, True), (7, 9, False))
TEXT_INDICES = [0, 1, 2, 3, 3, 3, 3, 4, 5]
IDS_A = [10, 11, 12, 99, 20, 21]
IDS_B = [10, 11, 12, 99, 30, 31]


def chain(ids, image=b"img1"):
    return keys.segment_chain(b"root", SEGMENTS, TEXT_INDICES, ids, [image], [(2, 2)])


def test_chain_shares_prefix_until_first_difference():
    a, b = chain(IDS_A), chain(IDS_B)
    assert [k for *_, k in a[:2]] == [k for *_, k in b[:2]]
    assert a[2][2] != b[2][2]
    assert chain(IDS_A, b"img2")[1][2] != a[1][2]  # different image: same text before it, new key after
    assert chain(IDS_A, b"img2")[0][2] == a[0][2]


def test_chain_depends_on_root():
    assert keys.segment_chain(b"other", SEGMENTS, TEXT_INDICES, IDS_A, [b"img1"], [(2, 2)])[0][2] != chain(IDS_A)[0][2]


def test_image_key_sees_pixels_and_size():
    a = Image.new("RGB", (32, 32), (1, 2, 3))
    b = Image.new("RGB", (32, 32), (1, 2, 4))
    assert keys.image_content_key(a) == keys.image_content_key(a.copy())
    assert keys.image_content_key(a) != keys.image_content_key(b)
    assert keys.image_content_key(a) != keys.image_content_key(a.resize((64, 16)))


def _kv(tokens, fill=0.0):
    return torch.full((2, 2, tokens, 1, 4), fill)


def test_store_match_insert_and_branching():
    store = PrefixKVStore(gpu_budget=1 << 30, cpu_budget=1 << 30)
    a, b = [k for *_, k in chain(IDS_A)], [k for *_, k in chain(IDS_B)]
    parent = None
    for key, (s, e, _) in zip(a, SEGMENTS):
        parent = store.insert(parent, key, _kv(e - s))
    assert len(store.match(a, torch.device("cpu"))) == 3
    matched = store.match(b, torch.device("cpu"))
    assert len(matched) == 2  # shares text + image, differs in the final text
    store.insert(matched[-1], b[2], _kv(2, 1.0))
    assert len(store.match(b, torch.device("cpu"))) == 3
    assert len(store.nodes) == 4


def test_store_drops_subtree_over_cpu_budget():
    one = _kv(4).numel() * 4
    store = PrefixKVStore(gpu_budget=0, cpu_budget=2 * one)
    a = [k for *_, k in chain(IDS_A)]
    n0 = store.insert(None, a[0], _kv(4))
    n1 = store.insert(n0, a[1], _kv(4))
    store.insert(n1, a[2], _kv(4))
    n0.last_used = 0  # oldest: dropping it takes its whole subtree
    store.enforce_budgets()
    assert store.match(a, torch.device("cpu")) == []
    assert not store.nodes


def test_rasterize_box_is_one_inside_zero_far_outside():
    spec = MaskSpec(boxes=[(64, 64, 128, 128)], feather_px=16)
    m = rasterize(spec, 256, 256).view(16, 16)
    assert m[4:8, 4:8].min() == 1  # cells 4..7 cover pixels 64..127
    assert m[0, 0] == 0 and m[15, 15] == 0
    assert 0 < m[3, 5] < 1  # feathered neighbor


def test_blend_keeps_source_outside_at_sigma_zero():
    x0 = torch.randn(1, 4, 3)
    state = MaskState(x0=x0, editable=torch.tensor([1.0, 0, 0, 1]).view(1, 4, 1), eps=torch.randn(1, 4, 3))
    out = state.blend(torch.zeros(1, 4, 3), 0.0)
    assert torch.equal(out[0, 1], x0[0, 1]) and torch.equal(out[0, 0], torch.zeros(3))


def test_request_payload_roundtrip():
    class Batch:
        extra = {"diffusers_kwargs": {"mt": {"store_output_as": "s/v1", "ref_latents": {"0": "s/v0"},
                                             "mask": {"boxes": [[0, 0, 32, 32]], "canvas_ref": 0}}}}

    mt = MTRequest.from_batch(Batch())
    assert mt.store_output_as == "s/v1" and mt.ref_latents == {0: "s/v0"}
    assert mt.mask.boxes == [(0.0, 0.0, 32.0, 32.0)]
    assert MTRequest.from_batch(object()).store_output_as is None
