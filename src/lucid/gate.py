"""Adaptive doubt gate: triage world-model disagreements into revise/adopt/ignore."""

from __future__ import annotations

from lucid.core import Prediction


class DoubtGate:
    """Spends the revision budget where the WM is confident, likely right, and stakes are real."""

    def __init__(self, theta: float, u_max: float = 0.4, impact_other: float = 0.4):
        self.theta = theta
        self.u_max = u_max
        self.impact_other = impact_other

    def decide(
        self, pred: Prediction, disputed: list[str], goals: dict[int, str]
    ) -> tuple[str, float]:
        """Returns (decision, doubt). decision in {"revise", "adopt", "ignore"}."""
        goal_vars = {"robot_zone"} | {f"box_{i}" for i in goals}
        impact = 1.0 if any(v in goal_vars for v in disputed) else self.impact_other
        doubt = (1 - pred.disagreement) * (0.5 * (1 - pred.validity_prob) + 0.5 * impact)
        if pred.disagreement > self.u_max:
            return "ignore", doubt
        return ("revise", doubt) if doubt >= self.theta else ("adopt", doubt)
