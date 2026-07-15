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


def frontier_plot(summary: dict, path) -> None:
    """Cost-accuracy frontier: extra LLM calls vs. ungated (x) against hallucination rate (y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = summary["ungated"]["calls_per_episode"]

    def xy(arm):
        s = summary[arm]
        return 100 * (s["calls_per_episode"] / base - 1), 100 * s["hallucinated_state_rate"]

    fig, ax = plt.subplots(figsize=(7, 5))
    adaptive = sorted((xy(a) for a in summary if a.startswith("adaptive_")), key=lambda p: p[0])
    if adaptive:
        ax.plot(*zip(*adaptive), "o-", color="tab:blue", label="adaptive doubt gate")
    for arm, marker, color in [
        ("ungated", "s", "tab:red"),
        ("always_check", "D", "tab:orange"),
        ("oracle", "*", "tab:green"),
    ]:
        if arm in summary:
            x, y = xy(arm)
            ax.scatter([x], [y], marker=marker, s=90, color=color, zorder=3, label=arm)
    ax.set(
        xlabel="extra LLM calls vs. ungated (%)",
        ylabel="hallucinated-state rate (%)",
        title="Hallucination vs. cost: the doubt-budget frontier",
    )
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
