"""LLM planner: cached LiteLLM calls, belief tracking, draft->check->revise loop."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import litellm

from lucid.core import HAND, Action, EnvConfig, State


@dataclass(frozen=True)
class Task:
    initial: State
    goals: dict[int, str]


def system_prompt(cfg: EnvConfig) -> str:
    zones = ", ".join(cfg.zones)
    return (
        f"You control a warehouse robot. Zones: {zones}. "
        f"Boxes are box_0..box_{cfg.n_boxes - 1}. The robot holds at most one box.\n"
        "Actions and preconditions:\n"
        '- {"kind": "move", "target": "<zone>"} — valid unless already in that zone.\n'
        '- {"kind": "pick", "target": <box index>} — valid iff the robot is in that box\'s zone '
        "and holds nothing.\n"
        '- {"kind": "place", "target": null} — puts the held box in the current zone; '
        "valid iff holding a box.\n"
        "Invalid actions do nothing.\n"
        "After each action you are told ONLY whether it was valid — never the state. "
        "Track the state yourself.\n"
        "Reply with EXACTLY one JSON object and nothing else:\n"
        '{"action": {"kind": "...", "target": ...}, '
        '"believed_next_state": {"robot_zone": "<zone>", "box_zones": ["<zone or hand>", ...]}}\n'
        f'"box_zones" lists all {cfg.n_boxes} boxes in index order; "hand" means held.'
    )


def user_message(task: Task, history: list[str], note: str | None) -> str:
    goals = ", ".join(f"box_{i} -> {z}" for i, z in sorted(task.goals.items()))
    lines = [f"Goals: {goals}", f"Initial state: {task.initial.to_json()}"]
    if history:
        lines.append("History:")
        lines.extend(history)
    if note:
        lines.append(f"Checker note: {note}")
    lines.append("Next action?")
    return "\n".join(lines)


def parse_reply(text: str, cfg: EnvConfig) -> tuple[Action, State]:
    """Strict parse of the model reply; raises ValueError/KeyError/TypeError on anything off."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    d = json.loads(m.group())
    a = d["action"]
    kind = a["kind"]
    if kind not in ("move", "pick", "place"):
        raise ValueError(f"bad action kind: {kind!r}")
    target = a.get("target")
    if kind == "move" and target not in cfg.zones:
        raise ValueError(f"bad move target: {target!r}")
    if kind == "pick" and not (
        isinstance(target, int) and not isinstance(target, bool) and 0 <= target < cfg.n_boxes
    ):
        raise ValueError(f"bad pick target: {target!r}")
    if kind == "place":
        target = None
    b = d["believed_next_state"]
    robot_zone = b["robot_zone"]
    box_zones = b["box_zones"]
    if robot_zone not in cfg.zones:
        raise ValueError(f"bad robot_zone: {robot_zone!r}")
    if len(box_zones) != cfg.n_boxes:
        raise ValueError(f"expected {cfg.n_boxes} box_zones, got {len(box_zones)}")
    if any(z != HAND and z not in cfg.zones for z in box_zones):
        raise ValueError(f"bad box_zones: {box_zones!r}")
    return Action(kind, target), State(robot_zone, tuple(box_zones))


class CachedLLM:
    """LiteLLM wrapper with a content-addressed disk cache. Cache hit -> zero network."""

    def __init__(self, model: str, cache_dir: Path):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def complete(self, messages: list[dict]) -> dict:
        """Returns {"text", "tokens_in", "tokens_out", "cache_hit"}."""
        key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "messages": messages, "temperature": 0}, sort_keys=True
            ).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text()) | {"cache_hit": True}
        resp = litellm.completion(model=self.model, messages=messages, temperature=0)
        out = {
            "text": resp.choices[0].message.content,
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens,
        }
        path.write_text(json.dumps(out))
        self.misses += 1
        return out | {"cache_hit": False}


def _blank_meta() -> dict:
    return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cache_hits": 0, "n_parse_retries": 0}


def _accumulate(meta: dict, other: dict) -> None:
    for k in meta:
        meta[k] += other[k]


class Planner:
    """Stepwise JSON-protocol planner with bounded retries on malformed replies."""

    def __init__(self, llm: CachedLLM, cfg: EnvConfig, max_parse_retries: int = 2):
        self.llm = llm
        self.cfg = cfg
        self.max_parse_retries = max_parse_retries

    def draft(
        self, task: Task, history: list[str], note: str | None
    ) -> tuple[Action | None, State | None, dict]:
        """One drafted (action, belief). (None, None, meta) means parse failure."""
        messages = [
            {"role": "system", "content": system_prompt(self.cfg)},
            {"role": "user", "content": user_message(task, history, note)},
        ]
        meta = _blank_meta()
        for attempt in range(1 + self.max_parse_retries):
            r = self.llm.complete(messages)
            meta["calls"] += 1
            meta["tokens_in"] += r["tokens_in"]
            meta["tokens_out"] += r["tokens_out"]
            meta["cache_hits"] += int(r["cache_hit"])
            try:
                action, belief = parse_reply(r["text"], self.cfg)
                return action, belief, meta
            except (ValueError, KeyError, TypeError) as e:
                meta["n_parse_retries"] = min(attempt + 1, self.max_parse_retries)
                messages += [
                    {"role": "assistant", "content": r["text"]},
                    {
                        "role": "user",
                        "content": f"Invalid reply ({e}). Reply with exactly one JSON object "
                        "in the required format.",
                    },
                ]
        return None, None, meta


class Checker:
    """Diffs the world model's prediction against the planner's asserted belief."""

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def check(
        self, prev_belief: State, action: Action, belief: State
    ) -> tuple[bool, str | None, State]:
        """Returns (ok, revision_note, wm_predicted_state). Never sees the true state."""
        pred = self.ensemble.predict(prev_belief, action)
        self.last_pred = pred
        wm_state = pred.point_next_state
        if wm_state == belief:
            return True, None, wm_state
        diffs = []
        if wm_state.robot_zone != belief.robot_zone:
            diffs.append(
                f"robot_zone: world model says {wm_state.robot_zone!r}, "
                f"you said {belief.robot_zone!r}"
            )
        for i, (w, b) in enumerate(zip(wm_state.box_zones, belief.box_zones)):
            if w != b:
                diffs.append(f"box_{i}: world model says {w!r}, you said {b!r}")
        note = (
            f"Your believed state disagrees with a trained world model "
            f"(action validity estimate {pred.validity_prob:.2f}). "
            + "; ".join(diffs)
            + ". Reconsider your action and belief."
        )
        return False, note, wm_state


def run_episode(
    env,
    planner: Planner,
    checker: Checker | None,
    task_seed: int,
    max_revisions: int = 2,
    gate=None,
) -> tuple[list[dict], bool]:
    """One episode; returns (per-step rows, success). Agent sees only valid flags."""
    from lucid.env import goals_met  # local import to avoid a cycle at module load

    state, goals = env.reset(task_seed)
    task = Task(state, goals)
    history: list[str] = []
    working_belief = state  # the checker's reference point; starts at the known initial state
    rows: list[dict] = []
    for step in range(env.cfg.max_steps):
        action, belief, meta = planner.draft(task, history, None)
        n_revisions = 0
        wm_state = None
        revision_parse_failure = False
        gate_decision = ""
        doubt = -1.0
        if action is None:
            rows.append(
                _row(
                    task_seed,
                    step,
                    env.state,
                    None,
                    None,
                    None,
                    False,
                    0,
                    meta,
                    True,
                    False,
                    "",
                    -1.0,
                )
            )
            history.append(f"step {step}: PARSE FAILURE - no action executed")
            continue
        if checker is not None:
            ok, note, wm_state = checker.check(working_belief, action, belief)
            decision = "revise"
            if not ok and gate is not None:
                from lucid.gate import disputed_vars

                decision, doubt = gate.decide(
                    checker.last_pred, disputed_vars(wm_state, belief), goals
                )
                gate_decision = decision
            if not ok and decision == "revise":
                while not ok and n_revisions < max_revisions:
                    n_revisions += 1
                    action2, belief2, meta2 = planner.draft(task, history, note)
                    _accumulate(meta, meta2)
                    if action2 is None:
                        revision_parse_failure = True  # stale disputed action executes below
                        break
                    action, belief = action2, belief2
                    ok, note, wm_state = checker.check(working_belief, action, belief)
                working_belief = belief if ok else wm_state
            elif not ok and decision == "adopt":
                working_belief = wm_state
            elif not ok:  # ignore
                working_belief = belief
            else:
                working_belief = belief
        else:
            working_belief = belief
        true_state, valid, done = env.step(action)
        history.append(f"step {step}: {action.to_json()} -> {'valid' if valid else 'invalid'}")
        if gate_decision == "adopt":
            history.append(
                f"note: your believed state after step {step} was corrected to {working_belief.to_json()}"
            )
        rows.append(
            _row(
                task_seed,
                step,
                true_state,
                belief,
                wm_state,
                action,
                valid,
                n_revisions,
                meta,
                False,
                revision_parse_failure,
                gate_decision,
                doubt,
            )
        )
        if done:
            break
    return rows, goals_met(env.state, goals)


def _row(
    episode_id,
    step,
    true_state,
    belief,
    wm_state,
    action,
    valid,
    n_revisions,
    meta,
    parse_failure,
    revision_parse_failure,
    gate_decision,
    doubt,
) -> dict:
    return {
        "episode_id": episode_id,
        "step": step,
        "true_state": true_state.to_json(),
        "believed_state": belief.to_json() if belief else None,
        "wm_predicted_state": wm_state.to_json() if wm_state else None,
        "action": action.to_json() if action else None,
        "valid": valid,
        "n_revisions": n_revisions,
        "n_parse_retries": meta["n_parse_retries"],
        "parse_failure": parse_failure,
        "revision_parse_failure": revision_parse_failure,
        "gate_decision": gate_decision,
        "doubt": doubt,
        "calls": meta["calls"],
        "cache_hits": meta["cache_hits"],
        "tokens_in": meta["tokens_in"],
        "tokens_out": meta["tokens_out"],
    }


def summarize(episodes: list[tuple[list[dict], bool]]) -> dict:
    """Aggregate metrics over (rows, success) pairs. The headline: hallucinated-state rate."""
    rows = [r for ep, _ in episodes for r in ep]
    scored = [r for r in rows if not r["parse_failure"]]
    hallucinated = [r for r in scored if r["believed_state"] != r["true_state"]]
    first_div = []
    for ep, _ in episodes:
        div = [
            r["step"]
            for r in ep
            if not r["parse_failure"] and r["believed_state"] != r["true_state"]
        ]
        if div:
            first_div.append(min(div))
    calls = sum(r["calls"] for r in rows)
    return {
        "n_episodes": len(episodes),
        "n_steps": len(rows),
        "hallucinated_state_rate": len(hallucinated) / len(scored) if scored else 0.0,
        "success_rate": sum(s for _, s in episodes) / len(episodes) if episodes else 0.0,
        "parse_failure_rate": sum(r["parse_failure"] for r in rows) / calls if calls else 0.0,
        "revision_parse_failure_rate": (
            sum(r["revision_parse_failure"] for r in rows) / len(rows) if rows else 0.0
        ),
        "invalid_action_rate": sum(not r["valid"] for r in scored) / len(scored) if scored else 0.0,
        "mean_first_divergence_step": (sum(first_div) / len(first_div)) if first_div else None,
        "calls_per_episode": calls / len(episodes) if episodes else 0.0,
        "revisions_per_step": sum(r["n_revisions"] for r in rows) / len(rows) if rows else 0.0,
        "adopt_rate": sum(r["gate_decision"] == "adopt" for r in rows) / len(rows) if rows else 0.0,
        "ignore_rate": sum(r["gate_decision"] == "ignore" for r in rows) / len(rows)
        if rows
        else 0.0,
        "cache_hit_rate": sum(r["cache_hits"] for r in rows) / calls if calls else 0.0,
        "tokens_in": sum(r["tokens_in"] for r in rows),
        "tokens_out": sum(r["tokens_out"] for r in rows),
    }


def run_arm(arm: str, run_cfg: dict, agent_cfg: dict) -> dict:
    """Run one arm over n_episodes; writes <out_dir>/<arm>.parquet, returns the summary dict."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lucid.env import WarehouseEnv
    from lucid.wm import load_ensemble

    env_cfg = EnvConfig(**run_cfg["env"])
    env = WarehouseEnv(env_cfg)
    llm = CachedLLM(agent_cfg["model"], Path(agent_cfg["cache_dir"]))
    planner = Planner(llm, env_cfg, agent_cfg["max_parse_retries"])
    checker = (
        Checker(load_ensemble(Path(run_cfg["wm_checkpoint"]))) if arm == "always_check" else None
    )
    episodes = []
    for i in range(run_cfg["n_episodes"]):
        episodes.append(
            run_episode(
                env, planner, checker, run_cfg["seed"] * 100_000 + i, agent_cfg["max_revisions"]
            )
        )
    out_dir = Path(run_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for ep, _ in episodes for r in ep]
    table = pa.table({k: [r[k] for r in rows] for k in rows[0]})
    pq.write_table(table, out_dir / f"{arm}.parquet")
    return summarize(episodes) | {
        "arm": arm,
        "llm_cache_hits": llm.hits,
        "llm_cache_misses": llm.misses,
    }
