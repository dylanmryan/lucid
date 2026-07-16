"""@grounded: add world-model grounding to any step-taking agent."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Protocol

from lucid.core import State
from lucid.gate import warehouse_goal_vars


class StateAdapter(Protocol):
    """The one domain-specific seam: flatten states to named variables."""

    def variables(self, state: Any) -> dict[str, str]: ...

    def goal_vars(self, state: Any, goals: Any) -> set[str]:
        """Which variable names are goal-relevant. Must tolerate goals=None."""
        ...


class WarehouseAdapter:
    def variables(self, state: State) -> dict[str, str]:
        return {"robot_zone": state.robot_zone} | {
            f"box_{i}": z for i, z in enumerate(state.box_zones)
        }

    def goal_vars(self, state: State, goals: dict[int, str] | None) -> set[str]:
        return warehouse_goal_vars(goals or {})


class GroundingSession:
    """Accumulates monitor-schema rows; .to_parquet writes what lucid-monitor replays."""

    def __init__(self, episode_id: int = 0):
        self.episode_id = episode_id
        self.rows: list[dict] = []
        self._step = 0

    def record(
        self,
        believed: dict[str, str],
        predicted: dict[str, str],
        action: Any,
        validity_prob: float,
        gate_decision: str,
        doubt: float,
        truth: dict[str, str] | None = None,
    ) -> None:
        # ponytail: in deployment there is no oracle — the WM prediction stands in for
        # ground truth in the monitor view unless the harness supplies the real thing.
        action_json = action.to_json() if hasattr(action, "to_json") else json.dumps(str(action))
        self.rows.append(
            {
                "episode_id": self.episode_id,
                "step": self._step,
                "true_state": json.dumps(truth if truth is not None else predicted),
                "believed_state": json.dumps(believed),
                "wm_predicted_state": json.dumps(predicted),
                "action": action_json,
                "valid": validity_prob >= 0.5,
                "n_revisions": 0,
                "n_parse_retries": 0,
                "parse_failure": False,
                "revision_parse_failure": False,
                "calls": 0,
                "cache_hits": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "gate_decision": gate_decision,
                "doubt": doubt,
            }
        )
        self._step += 1

    def to_parquet(self, path: Path | str) -> None:
        if not self.rows:
            raise ValueError("no rows recorded; nothing to write")
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({k: [r[k] for r in self.rows] for k in self.rows[0]}), path)


def grounded(transition_model, adapter: StateAdapter, gate=None, session=None, goals=None):
    """Wrap step(state, action) -> belief with world-model grounding.

    On a confident, gate-approved disagreement the wrapped function returns the world
    model's belief instead ("adopt"); "revise" is recorded but re-drafting stays the
    host agent's responsibility.
    """

    def decorate(step_fn):
        @functools.wraps(step_fn)
        def wrapped(state, action):
            belief = step_fn(state, action)
            pred = transition_model.predict(state, action)
            bv = adapter.variables(belief)
            pv = adapter.variables(pred.point_next_state)
            if not bv or not pv:
                raise ValueError("adapter.variables() must return a non-empty dict")
            disputed = [n for n in pv if pv[n] != bv.get(n)]
            decision, doubt, out = "", -1.0, belief
            if disputed and gate is not None:
                decision, doubt = gate.decide(pred, disputed, adapter.goal_vars(state, goals))
                if decision == "adopt":
                    out = pred.point_next_state
            if session is not None:
                session.record(bv, pv, action, pred.validity_prob, decision, doubt)
            return out

        return wrapped

    return decorate
