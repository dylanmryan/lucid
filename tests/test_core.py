import numpy as np
import pytest

from lucid.core import (
    HAND,
    Action,
    EnvConfig,
    State,
    decode_state,
    encode,
    encoding_dim,
    target_indices,
)


def test_config_zones_and_boxes():
    cfg = EnvConfig(n_boxes=2, n_shelves=2)
    assert cfg.zones == ("receiving", "staging", "packing", "shelf_a", "shelf_b")
    assert cfg.boxes == ("box_0", "box_1")


def test_state_held_and_json_round_trip():
    s = State("staging", ("shelf_a", HAND))
    assert s.held == 1
    assert State("staging", ("shelf_a", "packing")).held is None
    assert State.from_json(s.to_json()) == s


def test_action_json_round_trip():
    for a in [Action("move", "packing"), Action("pick", 0), Action("place")]:
        assert Action.from_json(a.to_json()) == a


CAPS = EnvConfig(n_boxes=4, n_shelves=3)


def test_encode_shape_dtype():
    s = State("receiving", ("shelf_a", "packing"))
    v = encode(s, Action("pick", 1), CAPS)
    assert v.shape == (encoding_dim(CAPS),)
    assert v.dtype == np.float32


def test_target_decode_round_trip():
    for s in [
        State("receiving", ("shelf_b", HAND)),
        State("packing", ("receiving", "staging")),
    ]:
        assert decode_state(target_indices(s, CAPS), 2, CAPS) == s


def test_encode_rejects_state_exceeding_caps():
    small = EnvConfig(n_boxes=2, n_shelves=1)
    s = State("receiving", ("shelf_a", "packing", "staging"))
    with pytest.raises(ValueError):
        encode(s, Action("place"), small)


def test_decode_state_rejects_inactive_padding():
    idx = target_indices(State("receiving", ("shelf_a",)), CAPS)  # has -1 padding
    with pytest.raises(ValueError):
        decode_state(idx, CAPS.n_boxes, CAPS)
