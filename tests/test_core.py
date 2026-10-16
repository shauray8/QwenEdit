"""CPU tests for the pieces that need no GPU or SGLang: keys, the K/V store, masks."""

import torch
from PIL import Image

from mtedit import keys
from mtedit.masking import MaskState, rasterize
from mtedit.request import MaskSpec, MTRequest

# a layout like build_layout's: text [0,3), image 2x2 [3,7), text [7,9)
SEGMENTS = ((0, 3, False), (3, 7, True), (7, 9, False))
TEXT_INDICES = [0, 1, 2, 3, 3, 3, 3, 4, 5]
IDS_A = [10, 11, 12, 99, 20, 21]
IDS_B = [10, 11, 12, 99, 30, 31]


def test_image_key_sees_pixels_and_size():
    a = Image.new("RGB", (32, 32), (1, 2, 3))
    b = Image.new("RGB", (32, 32), (1, 2, 4))
    assert keys.image_content_key(a) == keys.image_content_key(a.copy())
    assert keys.image_content_key(a) != keys.image_content_key(b)
    assert keys.image_content_key(a) != keys.image_content_key(a.resize((64, 16)))


def vids(ids, image=b"img1"):
    return keys.virtual_ids(b"root", SEGMENTS, TEXT_INDICES, ids, [image], [(2, 2)])


def test_virtual_ids_share_head_and_split_at_first_difference():
    a, b = vids(IDS_A), vids(IDS_B)
    assert (a[:7] == b[:7]).all()  # text + image identical
    assert a[7] != b[7] and a[8] != b[8]
    assert len(set(a.tolist())) == len(a)


def test_virtual_ids_image_block_matches_whole_or_not_at_all():
    a, c = vids(IDS_A), vids(IDS_A, b"img2")
    assert (a[:3] == c[:3]).all()
    assert not (a[3:7] == c[3:7]).any()  # every block position differs, so no partial block match
    assert not (a[7:] == c[7:]).any()


def test_sequence_ids_chain_over_image_runs():
    toks = [1, 2, 99, 99, 99, 3, 4]
    a = keys.sequence_ids(b"r", toks, [(2, 5, b"x")])
    b = keys.sequence_ids(b"r", toks, [(2, 5, b"y")])
    c = keys.sequence_ids(b"r", toks[:5] + [3, 5], [(2, 5, b"x")])
    assert (a[:2] == b[:2]).all() and not (a[2:] == b[2:]).any()
    assert (a[:6] == c[:6]).all() and a[6] != c[6]


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
