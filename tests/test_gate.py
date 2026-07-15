from lucid.core import HAND, Action, EnvConfig, Prediction, State
from lucid.gate import DoubtGate, OracleChecker, disputed_vars
from lucid.env import WarehouseEnv

CFG = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)


def _pred(disagreement, validity_prob):
    s = State("receiving", ("receiving", "shelf_a"))
    return Prediction(
        validity_prob=validity_prob,
        point_next_state=s,
        disagreement=disagreement,
        per_var_disagreement={},
        entropy=0.0,
        next_state_dist={},
    )


def test_ignore_when_wm_uncertain():
    gate = DoubtGate(theta=0.2, u_max=0.4)
    decision, doubt = gate.decide(_pred(0.6, 0.03), ["box_0"], {0: "shelf_a"})
    assert decision == "ignore"


def test_revise_on_confident_invalid_goal_relevant():
    gate = DoubtGate(theta=0.45)
    decision, doubt = gate.decide(_pred(0.05, 0.03), ["box_0"], {0: "shelf_a"})
    assert decision == "revise"
    # doubt = 0.95 * (0.5 * 0.97 + 0.5 * 1.0) ~= 0.936
    assert 0.9 < doubt < 0.95


def test_adopt_on_confident_valid_irrelevant():
    gate = DoubtGate(theta=0.45)
    decision, doubt = gate.decide(_pred(0.05, 0.9), ["box_1"], {0: "shelf_a"})
    assert decision == "adopt"
    # doubt = 0.95 * (0.5 * 0.1 + 0.5 * 0.4) ~= 0.2375
    assert 0.2 < doubt < 0.3


def test_robot_zone_is_always_goal_relevant():
    gate = DoubtGate(theta=0.45)
    decision, _ = gate.decide(_pred(0.05, 0.03), ["robot_zone"], {0: "shelf_a"})
    assert decision == "revise"


def test_theta_monotonicity():
    pred, disputed, goals = _pred(0.05, 0.5), ["box_0"], {0: "shelf_a"}
    decisions = [DoubtGate(theta=t).decide(pred, disputed, goals)[0] for t in (0.1, 0.5, 0.9)]
    # raising theta can only turn revise into adopt, never the reverse
    assert decisions == sorted(decisions, key=lambda d: d == "adopt")


def test_disputed_vars():
    a = State("receiving", ("receiving", "shelf_a"))
    b = State("staging", ("receiving", HAND))
    assert disputed_vars(a, b) == ["robot_zone", "box_1"]
    assert disputed_vars(a, a) == []


def test_oracle_checker_flags_exactly_true_divergence():
    env = WarehouseEnv(CFG)
    state, _ = env.reset(seed=3)
    oracle = OracleChecker(env)
    action = Action("move", next(z for z in CFG.zones if z != state.robot_zone))
    true_next = State(action.target, state.box_zones)
    ok, note, ref = oracle.check(state, action, true_next)
    assert ok and note is None and ref == true_next
    wrong = State(state.robot_zone, state.box_zones)  # believes the move didn't happen
    ok, note, ref = oracle.check(state, action, wrong)
    assert not ok and "robot_zone" in note and ref == true_next
