import json

from lucid.monitor import VarView, step_event, step_events

TRUE_S = '{"robot_zone": "staging", "box_zones": ["receiving", "shelf_a"]}'
LIE_S = '{"robot_zone": "packing", "box_zones": ["receiving", "shelf_a"]}'


def _row(step, believed, predicted=None, decision="", doubt=-1.0, parse_failure=False):
    return {
        "episode_id": 5,
        "step": step,
        "true_state": TRUE_S,
        "believed_state": believed,
        "wm_predicted_state": predicted,
        "action": '{"kind": "move", "target": "staging"}',
        "valid": True,
        "n_revisions": 0,
        "n_parse_retries": 0,
        "parse_failure": parse_failure,
        "revision_parse_failure": False,
        "calls": 1,
        "cache_hits": 0,
        "tokens_in": 10,
        "tokens_out": 5,
        "gate_decision": decision,
        "doubt": doubt,
    }


def test_step_event_grounded():
    ev = step_event(_row(0, TRUE_S))
    assert ev.grounding == 1.0
    assert [v.name for v in ev.variables] == ["robot_zone", "box_0", "box_1"]
    assert all(v.agree for v in ev.variables)
    assert ev.variables[0].predicted == "staging"  # no WM row -> truth stands in


def test_step_event_hallucinating():
    ev = step_event(_row(1, LIE_S, predicted=TRUE_S, decision="adopt", doubt=0.4))
    assert ev.grounding == 2 / 3  # robot_zone lies, both boxes honest
    assert ev.variables[0] == VarView("robot_zone", "packing", "staging", "staging", False)
    assert ev.gate_decision == "adopt" and ev.doubt == 0.4


def test_step_event_json_round_trip():
    ev = step_event(_row(0, TRUE_S))
    d = json.loads(ev.to_json())
    assert d["step"] == 0 and d["variables"][0]["name"] == "robot_zone"


def test_step_events_skips_parse_failures_and_sorts():
    rows = [_row(2, TRUE_S), _row(0, TRUE_S), _row(1, None, parse_failure=True)]
    evs = step_events(rows)
    assert [e.step for e in evs] == [0, 2]


def test_flat_dict_states_supported():
    flat = '{"position": "3", "fuel": "low"}'
    ev = step_event(_row(0, flat) | {"true_state": flat, "wm_predicted_state": None})
    assert [v.name for v in ev.variables] == ["position", "fuel"]
    assert ev.grounding == 1.0
