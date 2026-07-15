import json
import random
from types import SimpleNamespace

import pytest

import lucid.agent as agent_mod
from lucid.agent import (
    CachedLLM,
    Checker,
    Planner,
    Task,
    parse_reply,
    run_arm,
    run_episode,
    summarize,
    system_prompt,
    user_message,
)
from lucid.core import HAND, Action, EnvConfig, Prediction, State
from lucid.env import WarehouseEnv, make_task, solve, transition

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


def fake_completion_factory(replies: list[str], counter: dict):
    def fake(model, messages, temperature):
        counter["calls"] += 1
        text = replies[min(counter["calls"] - 1, len(replies) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    return fake


def test_cached_llm_caches(tmp_path, monkeypatch):
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(["hi"], counter))
    llm = CachedLLM("test-model", tmp_path)
    messages = [{"role": "user", "content": "x"}]
    r1 = llm.complete(messages)
    r2 = llm.complete(messages)
    assert counter["calls"] == 1  # second call served from disk
    assert r1["text"] == r2["text"] == "hi"
    assert r1["cache_hit"] is False and r2["cache_hit"] is True
    assert (llm.hits, llm.misses) == (1, 1)
    r3 = llm.complete([{"role": "user", "content": "different"}])
    assert counter["calls"] == 2 and r3["cache_hit"] is False


TASK = Task(State("receiving", ("receiving", "shelf_a")), {0: "shelf_b"})


def _planner(tmp_path, monkeypatch, replies):
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    return Planner(CachedLLM("test-model", tmp_path), CFG, max_parse_retries=2), counter


def test_planner_good_first_try(tmp_path, monkeypatch):
    planner, counter = _planner(tmp_path, monkeypatch, [GOOD])
    action, belief, meta = planner.draft(TASK, [], None)
    assert action == Action("pick", 1)
    assert belief.robot_zone == "shelf_a"
    assert meta["calls"] == 1 and meta["n_parse_retries"] == 0


def test_planner_recovers_from_malformed(tmp_path, monkeypatch):
    planner, counter = _planner(tmp_path, monkeypatch, ["garbage", GOOD])
    action, belief, meta = planner.draft(TASK, [], None)
    assert action == Action("pick", 1)
    assert meta["calls"] == 2 and meta["n_parse_retries"] == 1


def test_planner_parse_failure_after_retries(tmp_path, monkeypatch):
    planner, counter = _planner(tmp_path, monkeypatch, ["nope", "still nope", "never"])
    action, belief, meta = planner.draft(TASK, [], None)
    assert action is None and belief is None
    assert meta["calls"] == 3 and meta["n_parse_retries"] == 2


class FakeEnsemble:
    """Stands in for lucid.wm.Ensemble: predict() returns a canned Prediction."""

    def __init__(self, next_state, validity_prob=0.9):
        self.next_state = next_state
        self.validity_prob = validity_prob

    def predict(self, state, action):
        return Prediction(
            validity_prob=self.validity_prob,
            point_next_state=self.next_state,
            disagreement=0.0,
            per_var_disagreement={},
            entropy=0.0,
            next_state_dist={},
        )


def test_checker_ok_on_agreement():
    s = State("shelf_a", ("receiving", HAND))
    ok, note, wm_state = Checker(FakeEnsemble(s)).check(
        State("shelf_a", ("receiving", "shelf_a")), Action("pick", 1), s
    )
    assert ok and note is None and wm_state == s


def test_checker_note_on_disagreement():
    wm = State("shelf_a", ("receiving", "shelf_a"))  # pick failed per WM
    belief = State("shelf_a", ("receiving", HAND))
    ok, note, wm_state = Checker(FakeEnsemble(wm, validity_prob=0.03)).check(
        State("shelf_a", ("receiving", "shelf_a")), Action("pick", 1), belief
    )
    assert not ok and wm_state == wm
    assert "box_1" in note and "0.03" in note


def scripted_replies(cfg, seed, lie_at_step=None):
    """Build the exact reply sequence a perfect (or once-lying) agent would emit."""
    state, goals = make_task(cfg, random.Random(seed))
    replies = []
    cur = state
    for step, action in enumerate(solve(state, goals, cfg)):
        nxt, _ = transition(cur, action, cfg)
        believed = nxt
        if step == lie_at_step:
            wrong_zone = next(z for z in cfg.zones if z != nxt.robot_zone)
            believed = State(wrong_zone, nxt.box_zones)
        replies.append(
            json.dumps(
                {
                    "action": json.loads(action.to_json()),
                    "believed_next_state": json.loads(believed.to_json()),
                }
            )
        )
        cur = nxt
    return replies


def test_run_episode_perfect_agent(tmp_path, monkeypatch):
    cfg = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)
    replies = scripted_replies(cfg, seed=7)
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    planner = Planner(CachedLLM("test-model", tmp_path), cfg)
    rows, success = run_episode(WarehouseEnv(cfg), planner, None, task_seed=7, max_revisions=2)
    assert success is True
    assert all(r["believed_state"] == r["true_state"] for r in rows)
    assert rows[0]["parse_failure"] is False
    expected_keys = {
        "episode_id",
        "step",
        "true_state",
        "believed_state",
        "wm_predicted_state",
        "action",
        "valid",
        "n_revisions",
        "n_parse_retries",
        "parse_failure",
        "revision_parse_failure",
        "gate_decision",
        "doubt",
        "calls",
        "cache_hits",
        "tokens_in",
        "tokens_out",
    }
    assert set(rows[0]) == expected_keys


def test_run_episode_hallucination_counted(tmp_path, monkeypatch):
    cfg = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)
    replies = scripted_replies(cfg, seed=7, lie_at_step=1)
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    planner = Planner(CachedLLM("test-model", tmp_path), cfg)
    rows, success = run_episode(WarehouseEnv(cfg), planner, None, task_seed=7, max_revisions=2)
    summary = summarize([(rows, success)])
    assert 0 < summary["hallucinated_state_rate"] < 1
    assert summary["n_episodes"] == 1
    assert summary["mean_first_divergence_step"] == 1


def test_summarize_perfect(tmp_path, monkeypatch):
    cfg = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)
    replies = scripted_replies(cfg, seed=7)
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    planner = Planner(CachedLLM("test-model", tmp_path), cfg)
    rows, success = run_episode(WarehouseEnv(cfg), planner, None, task_seed=7, max_revisions=2)
    s = summarize([(rows, success)])
    assert s["hallucinated_state_rate"] == 0.0
    assert s["success_rate"] == 1.0
    assert s["parse_failure_rate"] == 0.0
    assert s["mean_first_divergence_step"] is None


def test_run_arm_writes_outputs(tmp_path, monkeypatch):
    replies = scripted_replies(CFG, seed=11 * 100_000) + scripted_replies(
        CFG, seed=11 * 100_000 + 1
    )
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
        "n_episodes": 2,
        "seed": 11,
        "wm_checkpoint": "unused",
        "out_dir": str(tmp_path / "out"),
    }
    summary = run_arm("ungated", run_cfg, agent_cfg)
    assert (tmp_path / "out" / "ungated.parquet").exists()
    assert summary["n_episodes"] == 2
    assert summary["arm"] == "ungated"


def test_run_agent_main_rejects_unknown_arm(tmp_path, monkeypatch):
    import sys

    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[agent]\nmodel = "test-model"\nmax_parse_retries = 2\nmax_revisions = 2\n'
        f'cache_dir = "{tmp_path}/cache"\n'
        "[run]\nenv = { n_boxes = 2, n_shelves = 2, max_steps = 30 }\n"
        'n_episodes = 1\nseed = 1\narms = ["ungated", "always_check"]\n'
        f'wm_checkpoint = "unused"\nout_dir = "{tmp_path}/out"\n'
    )
    monkeypatch.setattr(sys, "argv", ["lucid-run-agent", "--config", str(cfg), "--arm", "typo"])
    from lucid.cli import run_agent_main

    with pytest.raises(SystemExit):
        run_agent_main()


class PerfectEnsemble:
    """A world model that is exactly right: predicts via the env's true dynamics."""

    def __init__(self, cfg):
        self.cfg = cfg

    def predict(self, state, action):
        nxt, valid = transition(state, action, self.cfg)
        return Prediction(
            validity_prob=1.0 if valid else 0.0,
            point_next_state=nxt,
            disagreement=0.0,
            per_var_disagreement={},
            entropy=0.0,
            next_state_dist={},
        )


def test_checker_revision_corrects_lie(tmp_path, monkeypatch):
    correct = scripted_replies(CFG, seed=7)
    lied = scripted_replies(CFG, seed=7, lie_at_step=0)
    replies = [lied[0], correct[0]] + correct[1:]  # lie once, corrected on revision
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    planner = Planner(CachedLLM("test-model", tmp_path), CFG)
    checker = Checker(PerfectEnsemble(CFG))
    rows, success = run_episode(WarehouseEnv(CFG), planner, checker, task_seed=7, max_revisions=2)
    assert success is True
    assert rows[0]["n_revisions"] == 1
    assert rows[0]["calls"] == 2
    assert rows[0]["believed_state"] == rows[0]["true_state"]  # revision fixed the lie
    assert all(r["n_revisions"] == 0 for r in rows[1:])


def test_checker_adopts_wm_belief_after_exhausted_revisions(tmp_path, monkeypatch):
    correct = scripted_replies(CFG, seed=7)
    lied = scripted_replies(CFG, seed=7, lie_at_step=0)
    # At step 0: lie 3 times (exhausts revisions); at step 1+: one revision per step
    replies = [lied[0], lied[0], lied[0], lied[1], correct[1], correct[2], correct[3]]
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    planner = Planner(CachedLLM("test-model", tmp_path), CFG)
    checker = Checker(PerfectEnsemble(CFG))
    rows, success = run_episode(WarehouseEnv(CFG), planner, checker, task_seed=7, max_revisions=2)
    assert rows[0]["n_revisions"] == 2
    assert rows[0]["calls"] == 3
    assert rows[0]["believed_state"] != rows[0]["true_state"]  # lie was not corrected
    # After WM adoption at step 0, step 1 starts with correct working_belief
    # but planner still outputs a lie (step 1 with lied[1]), which gets corrected
    assert rows[1]["n_revisions"] == 1
    assert success is True


def test_revision_parse_failure_flagged(tmp_path, monkeypatch):
    correct = scripted_replies(CFG, seed=7)
    lied = scripted_replies(CFG, seed=7, lie_at_step=0)
    # Step 0: lie parses fine, checker disagrees, revision re-draft is garbage
    # 3 times (exhausts parse retries) -> stale disputed action executes anyway.
    replies = [lied[0], "garbage", "garbage", "garbage"] + correct[1:]
    counter = {"calls": 0}
    monkeypatch.setattr(agent_mod.litellm, "completion", fake_completion_factory(replies, counter))
    planner = Planner(CachedLLM("test-model", tmp_path), CFG)
    checker = Checker(PerfectEnsemble(CFG))
    rows, success = run_episode(WarehouseEnv(CFG), planner, checker, task_seed=7, max_revisions=2)
    assert rows[0]["parse_failure"] is False  # an action did execute
    assert rows[0]["revision_parse_failure"] is True
    assert rows[0]["n_revisions"] == 1
    assert rows[0]["calls"] == 4  # 1 draft + 3 wasted revision calls, now visible
    assert all(r["revision_parse_failure"] is False for r in rows[1:])
    summary = summarize([(rows, success)])
    assert summary["revision_parse_failure_rate"] == 1 / len(rows)
