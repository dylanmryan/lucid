import pytest

from lucid.agent import Task, parse_reply, system_prompt, user_message
from lucid.core import HAND, Action, EnvConfig, State

CFG = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)

GOOD = (
    '{"action": {"kind": "pick", "target": 1}, '
    '"believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving", "hand"]}}'
)


def test_parse_reply_good():
    action, belief = parse_reply(GOOD, CFG)
    assert action == Action("pick", 1)
    assert belief == State("shelf_a", ("receiving", HAND))


def test_parse_reply_tolerates_surrounding_prose():
    action, _ = parse_reply(f"Sure! Here you go:\n{GOOD}\nDone.", CFG)
    assert action.kind == "pick"


def test_parse_reply_rejects_bad_input():
    bad = [
        "no json here",
        '{"action": {"kind": "jump", "target": 1}, "believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving", "hand"]}}',
        '{"action": {"kind": "move", "target": "nowhere"}, "believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving", "hand"]}}',
        '{"action": {"kind": "pick", "target": 9}, "believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving", "hand"]}}',
        '{"action": {"kind": "pick", "target": "box_1"}, "believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving", "hand"]}}',
        '{"action": {"kind": "pick", "target": true}, "believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving", "hand"]}}',
        '{"action": {"kind": "place"}, "believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving"]}}',
        '{"action": {"kind": "place"}, "believed_next_state": {"robot_zone": "mars", "box_zones": ["receiving", "hand"]}}',
        '{"believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["receiving", "hand"]}}',
    ]
    for text in bad:
        with pytest.raises((ValueError, KeyError, TypeError)):
            parse_reply(text, CFG)


def test_place_target_normalized_to_none():
    text = (
        '{"action": {"kind": "place", "target": "shelf_a"}, '
        '"believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["shelf_a", "receiving"]}}'
    )
    action, _ = parse_reply(text, CFG)
    assert action == Action("place")


def test_system_prompt_mentions_all_zones_and_boxes():
    sp = system_prompt(CFG)
    for z in CFG.zones:
        assert z in sp
    assert "box_1" in sp
    assert "hand" in sp


def test_user_message_includes_history_and_note():
    task = Task(State("receiving", ("receiving", "shelf_a")), {0: "shelf_b"})
    msg = user_message(task, ['step 0: {"kind": "pick", "target": 0} -> valid'], "wm disagrees")
    assert "box_0 -> shelf_b" in msg
    assert "pick" in msg and "-> valid" in msg
    assert "Checker note: wm disagrees" in msg
