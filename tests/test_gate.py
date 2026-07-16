from lucid.core import HAND, Action, EnvConfig, Prediction, State
from lucid.gate import DoubtGate, OracleChecker, disputed_vars
from lucid.env import WarehouseEnv
import lucid.agent as agent_mod
from lucid.agent import CachedLLM, Checker, Planner, run_episode, summarize

from tests.test_agent import PerfectEnsemble, fake_completion_factory, scripted_replies

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
    decision, doubt = gate.decide(_pred(0.6, 0.03), ["box_0"], {"robot_zone", "box_0"})
    assert decision == "ignore"


def test_revise_on_confident_invalid_goal_relevant():
    gate = DoubtGate(theta=0.45)
    decision, doubt = gate.decide(_pred(0.05, 0.03), ["box_0"], {"robot_zone", "box_0"})
    assert decision == "revise"
    # doubt = 0.95 * (0.5 * 0.97 + 0.5 * 1.0) ~= 0.936
    assert 0.9 < doubt < 0.95


def test_adopt_on_confident_valid_irrelevant():
    gate = DoubtGate(theta=0.45)
    decision, doubt = gate.decide(_pred(0.05, 0.9), ["box_1"], {"robot_zone", "box_0"})
    assert decision == "adopt"
    # doubt = 0.95 * (0.5 * 0.1 + 0.5 * 0.4) ~= 0.2375
    assert 0.2 < doubt < 0.3


def test_robot_zone_is_always_goal_relevant():
    gate = DoubtGate(theta=0.45)
    decision, _ = gate.decide(_pred(0.05, 0.03), ["robot_zone"], {"robot_zone", "box_0"})
    assert decision == "revise"


def test_theta_monotonicity():
    pred, disputed, goals = _pred(0.05, 0.5), ["box_0"], {"robot_zone", "box_0"}
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


def _gated_episode(tmp_path, monkeypatch, gate, lie=True):
    replies_correct = scripted_replies(CFG, seed=7)
    lied = scripted_replies(CFG, seed=7, lie_at_step=0)
    replies = ([lied[0]] + replies_correct[1:]) if lie else replies_correct
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    planner = Planner(CachedLLM("test-model", tmp_path), CFG)
    checker = Checker(PerfectEnsemble(CFG))
    return run_episode(WarehouseEnv(CFG), planner, checker, task_seed=7, max_revisions=2, gate=gate)


def test_gate_adopt_records_decision_and_corrects_tracking(tmp_path, monkeypatch):
    from lucid.gate import DoubtGate

    rows, success = _gated_episode(tmp_path, monkeypatch, DoubtGate(theta=1.01))  # never revise
    assert rows[0]["gate_decision"] == "adopt"
    assert rows[0]["doubt"] >= 0.0
    assert rows[0]["n_revisions"] == 0
    assert rows[0]["believed_state"] != rows[0]["true_state"]  # assertion still counted
    assert all(r["gate_decision"] == "" for r in rows[1:])
    assert success is True


def test_gate_ignore_leaves_agent_alone(tmp_path, monkeypatch):
    from lucid.gate import DoubtGate

    rows, _ = _gated_episode(
        tmp_path, monkeypatch, DoubtGate(theta=1.01, u_max=-1.0)
    )  # always ignore
    assert rows[0]["gate_decision"] == "ignore"
    assert rows[0]["n_revisions"] == 0


def test_gate_revise_matches_always_check(tmp_path, monkeypatch):
    from lucid.gate import DoubtGate

    rows, _ = _gated_episode(tmp_path, monkeypatch, DoubtGate(theta=-1.0))  # always revise
    assert rows[0]["gate_decision"] == "revise"
    assert rows[0]["n_revisions"] >= 1


def test_no_gate_rows_have_empty_decision(tmp_path, monkeypatch):
    rows, _ = _gated_episode(tmp_path, monkeypatch, None, lie=False)
    assert all(r["gate_decision"] == "" and r["doubt"] == -1.0 for r in rows)


def test_summarize_gate_rates(tmp_path, monkeypatch):
    from lucid.gate import DoubtGate

    rows, success = _gated_episode(tmp_path, monkeypatch, DoubtGate(theta=1.01))
    s = summarize([(rows, success)])
    assert s["adopt_rate"] > 0.0
    assert s["ignore_rate"] == 0.0


def test_run_arm_adaptive_and_oracle(tmp_path, monkeypatch):
    from lucid.agent import run_arm

    replies = scripted_replies(CFG, seed=11 * 100_000)
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    agent_cfg = {
        "model": "test-model",
        "max_parse_retries": 2,
        "max_revisions": 2,
        "cache_dir": str(tmp_path / "cache"),
    }
    run_cfg = {
        "env": {"n_boxes": 2, "n_shelves": 2, "max_steps": 30},
        "n_episodes": 1,
        "seed": 11,
        "wm_checkpoint": "unused-for-oracle",
        "out_dir": str(tmp_path / "out"),
    }
    gate_cfg = {"u_max": 0.4, "impact_other": 0.4, "thetas": {"adaptive_mid": 0.45}}
    s = run_arm("oracle", run_cfg, agent_cfg, gate_cfg)
    assert s["arm"] == "oracle"
    assert (tmp_path / "out" / "oracle.parquet").exists()


def test_run_arm_adaptive_builds_gate(tmp_path, monkeypatch):
    """The adaptive path loads a WM checker and a DoubtGate at the arm's theta."""
    import lucid.wm as wm_mod
    from lucid.agent import run_arm

    replies = scripted_replies(CFG, seed=11 * 100_000)
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    monkeypatch.setattr(wm_mod, "load_ensemble", lambda path: PerfectEnsemble(CFG))
    agent_cfg = {
        "model": "test-model",
        "max_parse_retries": 2,
        "max_revisions": 2,
        "cache_dir": str(tmp_path / "cache"),
    }
    run_cfg = {
        "env": {"n_boxes": 2, "n_shelves": 2, "max_steps": 30},
        "n_episodes": 1,
        "seed": 11,
        "wm_checkpoint": "unused-patched",
        "out_dir": str(tmp_path / "out"),
    }
    gate_cfg = {"u_max": 0.4, "impact_other": 0.4, "thetas": {"adaptive_mid": 0.45}}
    s = run_arm("adaptive_mid", run_cfg, agent_cfg, gate_cfg)
    assert s["arm"] == "adaptive_mid"
    assert (tmp_path / "out" / "adaptive_mid.parquet").exists()


def test_frontier_plot_writes_png(tmp_path):
    from lucid.gate import frontier_plot

    summary = {
        "ungated": {"calls_per_episode": 17.6, "hallucinated_state_rate": 0.136},
        "always_check": {"calls_per_episode": 19.8, "hallucinated_state_rate": 0.012},
        "adaptive_lo": {"calls_per_episode": 19.0, "hallucinated_state_rate": 0.02},
        "adaptive_mid": {"calls_per_episode": 18.3, "hallucinated_state_rate": 0.04},
        "adaptive_hi": {"calls_per_episode": 17.9, "hallucinated_state_rate": 0.08},
        "oracle": {"calls_per_episode": 18.0, "hallucinated_state_rate": 0.005},
    }
    out = tmp_path / "frontier.png"
    frontier_plot(summary, out)
    assert out.exists()


def test_warehouse_goal_vars():
    from lucid.gate import warehouse_goal_vars

    assert warehouse_goal_vars({0: "shelf_a", 2: "shelf_b"}) == {"robot_zone", "box_0", "box_2"}
