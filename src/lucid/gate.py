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


def disputed_vars(expected, belief) -> list[str]:
    """Variable names where two states differ: 'robot_zone', 'box_0', ..."""
    out = ["robot_zone"] if expected.robot_zone != belief.robot_zone else []
    out += [
        f"box_{i}" for i, (w, b) in enumerate(zip(expected.box_zones, belief.box_zones)) if w != b
    ]
    return out


class OracleChecker:
    """Ground-truth checker: flags exactly the true divergences. Upper bound — peeks at truth."""

    def __init__(self, env):
        self.env = env

    def check(self, prev_belief, action, belief):
        """Same interface as agent.Checker.check; ignores prev_belief, reads the env's true state."""
        from lucid.env import transition

        true_next, _ = transition(self.env.state, action, self.env.cfg)
        if true_next == belief:
            return True, None, true_next
        diffs = "; ".join(
            f"{v}: expected differs from your belief" for v in disputed_vars(true_next, belief)
        )
        note = (
            f"Your believed state is wrong. Disputed: {diffs}. Reconsider your action and belief."
        )
        return False, note, true_next
