"""Live grounding monitor: replay episode logs as belief-vs-reality streams."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VarView:
    name: str
    believed: str
    predicted: str
    truth: str
    agree: bool


@dataclass(frozen=True)
class StepEvent:
    """Wire format the monitor streams; one per executed step."""

    step: int
    action: str
    valid: bool
    variables: list[VarView]
    grounding: float  # fraction of variables where belief == truth
    gate_decision: str
    doubt: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _to_vars(state_json: str) -> dict[str, str]:
    """Flatten a logged state JSON to named variables (warehouse State or flat dict)."""
    d = json.loads(state_json)
    if "box_zones" in d:
        return {"robot_zone": d["robot_zone"]} | {
            f"box_{i}": z for i, z in enumerate(d["box_zones"])
        }
    return {str(k): str(v) for k, v in d.items()}


def step_event(row: dict) -> StepEvent:
    truth = _to_vars(row["true_state"])
    believed = _to_vars(row["believed_state"]) if row["believed_state"] else truth
    predicted = _to_vars(row["wm_predicted_state"]) if row["wm_predicted_state"] else truth
    variables = [
        VarView(n, believed.get(n, "?"), predicted.get(n, "?"), t, believed.get(n) == t)
        for n, t in truth.items()
    ]
    return StepEvent(
        step=row["step"],
        action=row["action"] or "",
        valid=row["valid"],
        variables=variables,
        grounding=sum(v.agree for v in variables) / len(variables),
        gate_decision=row["gate_decision"],
        doubt=row["doubt"],
    )


def step_events(rows: list[dict]) -> list[StepEvent]:
    """Ordered events for one episode; parse-failure rows carry no belief and are skipped."""
    live = [r for r in rows if not r["parse_failure"]]
    return [step_event(r) for r in sorted(live, key=lambda r: r["step"])]
