import json
from dataclasses import dataclass

import pyarrow.parquet as pq
import pytest

from lucid.core import EnvConfig, Prediction, State
from lucid.gate import DoubtGate
from lucid.grounded import GroundingSession, WarehouseAdapter, grounded

CFG = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)


class FixedModel:
    """A transition model that always predicts the given state, confidently."""

    def __init__(self, next_state, validity_prob=0.95, disagreement=0.05):
        self.next_state = next_state
        self.validity_prob = validity_prob
        self.disagreement = disagreement

    def predict(self, state, action):
        return Prediction(
            validity_prob=self.validity_prob,
            point_next_state=self.next_state,
            disagreement=self.disagreement,
            per_var_disagreement={},
            entropy=0.0,
            next_state_dist={},
        )


def test_grounded_warehouse_adopts_on_lie(tmp_path):
    truth_next = State("staging", ("receiving", "shelf_a"))
    session = GroundingSession()
    gate = DoubtGate(theta=1.01)  # never revise -> adopt on any confident disagreement

    @grounded(
        FixedModel(truth_next), WarehouseAdapter(), gate=gate, session=session, goals={0: "shelf_a"}
    )
    def step(state, action):
        return State("packing", ("receiving", "shelf_a"))  # lies about robot_zone

    out = step(State("receiving", ("receiving", "shelf_a")), None)
    assert out == truth_next  # adopt returned the WM belief
    assert session.rows[0]["gate_decision"] == "adopt"
    assert json.loads(session.rows[0]["believed_state"])["robot_zone"] == "packing"
    session.to_parquet(tmp_path / "wrapped.parquet")
    assert pq.read_table(tmp_path / "wrapped.parquet").num_rows == 1


def test_grounded_no_disagreement_records_clean_row():
    truth_next = State("staging", ("receiving", "shelf_a"))
    session = GroundingSession()

    @grounded(
        FixedModel(truth_next),
        WarehouseAdapter(),
        gate=DoubtGate(theta=0.0),
        session=session,
        goals={},
    )
    def step(state, action):
        return truth_next

    out = step(State("receiving", ("receiving", "shelf_a")), None)
    assert out == truth_next
    assert session.rows[0]["gate_decision"] == "" and session.rows[0]["doubt"] == -1.0


def test_empty_adapter_variables_rejected():
    class EmptyAdapter:
        def variables(self, state):
            return {}

        def goal_vars(self, state, goals):
            return set()

    truth_next = State("staging", ("receiving", "shelf_a"))

    @grounded(FixedModel(truth_next), EmptyAdapter(), session=GroundingSession())
    def step(state, action):
        return truth_next

    with pytest.raises(ValueError, match="variables"):
        step(State("receiving", ("receiving", "shelf_a")), None)


# --- the generality test: a domain the codebase has never seen ---


@dataclass(frozen=True)
class RoverState:
    position: str
    fuel: str


class RoverAdapter:
    def variables(self, state):
        return {"position": state.position, "fuel": state.fuel}

    def goal_vars(self, state, goals):
        return {"position"}


def test_grounded_generalizes_to_foreign_domain(tmp_path):
    wm_belief = RoverState("crater", "low")
    session = GroundingSession()
    gate = DoubtGate(theta=1.01)

    @grounded(FixedModel(wm_belief), RoverAdapter(), gate=gate, session=session, goals=None)
    def step(state, action):
        return RoverState("ridge", "low")  # disagrees on position (goal-relevant)

    out = step(RoverState("base", "full"), "drive")
    assert out == wm_belief  # gate adopted the model's belief
    row = session.rows[0]
    assert row["gate_decision"] == "adopt"
    assert json.loads(row["believed_state"]) == {"position": "ridge", "fuel": "low"}
    # rows replay through the monitor's decoder
    from lucid.monitor import step_event

    ev = step_event(row)
    assert [v.name for v in ev.variables] == ["position", "fuel"]
