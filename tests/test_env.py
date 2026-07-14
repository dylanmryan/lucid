from lucid.core import HAND, Action, EnvConfig, State
from lucid.env import all_actions, is_valid, transition, valid_actions

CFG = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)
S = State("receiving", ("receiving", "shelf_a"))


def test_preconditions():
    assert is_valid(S, Action("pick", 0), CFG)
    assert not is_valid(S, Action("pick", 1), CFG)  # box elsewhere
    assert not is_valid(S, Action("move", "receiving"), CFG)  # already there
    assert is_valid(S, Action("move", "shelf_a"), CFG)
    assert not is_valid(S, Action("place"), CFG)  # empty hands
    held = State("receiving", (HAND, "shelf_a"))
    assert not is_valid(held, Action("pick", 1), CFG)  # hands full
    assert is_valid(held, Action("place"), CFG)


def test_invalid_is_noop():
    ns, valid = transition(S, Action("place"), CFG)
    assert not valid
    assert ns == S


def test_pick_move_place():
    s1, v1 = transition(S, Action("pick", 0), CFG)
    assert v1 and s1.box_zones[0] == HAND and s1.held == 0
    s2, _ = transition(s1, Action("move", "shelf_b"), CFG)
    s3, v3 = transition(s2, Action("place"), CFG)
    assert v3 and s3.box_zones[0] == "shelf_b" and s3.held is None


def test_action_enumeration():
    assert len(all_actions(CFG)) == 5 + 2 + 1  # moves + picks + place
    assert all(is_valid(S, a, CFG) for a in valid_actions(S, CFG))
    assert Action("pick", 0) in valid_actions(S, CFG)
