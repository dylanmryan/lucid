# Lucid — W3 Design: Adaptive Doubt-Budget Gate and the Cost-Accuracy Frontier

**Date:** 2026-07-15
**Status:** Approved
**Scope:** The W3 slice: the adaptive gate that triages world-model disagreements, the oracle upper bound, the six-arm evaluation, and the cost-accuracy frontier chart. The live monitor and `@grounded` wrapper remain W4.

## Context and the W2 reframe

W2 measured: ungated Haiku 4.5 hallucinates its asserted state on 13.6% of steps; revising on every WM disagreement (`always_check`) cuts that to 1.2% at +12.4% LLM calls, because disagreements only occur on ~14% of steps.

This changes the W3 framing from the original brief. The brief assumed *checks* are the cost to ration; in Lucid a WM check is a local CPU forward pass — free. The real economy is **what to do about a disagreement**:

- **revise** — pay an LLM call; the agent can change the about-to-execute action,
- **adopt** — free: the WM's prediction becomes the tracked working belief and a correction note is appended to history (fixes future drafts, not the current action),
- **ignore** — when the WM is self-uncertain (novel state), don't let a guessing checker overrule the agent.

The doubt budget is therefore **revision calls**, and the adaptive gate spends them where the WM is confident, the stakes are real, and a revision can still change the outcome.

Decisions made during brainstorming:

- Budget = revision calls (not checks). Gate decides revise / adopt / ignore per disagreement.
- Scalar doubt score with a single threshold θ (approach A); no learned gate, no per-episode hard quota.
- Hard eval budget: **≤ $14 of new API spend**, enforced operationally (arms run one at a time, real token spend checked between arms; remaining arms drop from 100 to 60 episodes if the cap approaches).
- No DuckDB (parquet + summary dict still suffice).

## The gate — `src/lucid/gate.py`

```
DoubtGate(theta: float, u_max: float = 0.4)
    .decide(pred: Prediction, disputed: list[str], goals: dict[int, str])
        -> "revise" | "adopt" | "ignore"
```

- `pred.disagreement > u_max` → **ignore** (WM self-uncertain; it may be the hallucinating party).
- Otherwise `doubt = (1 - pred.disagreement) * (0.5 * (1 - pred.validity_prob) + 0.5 * impact)` where `impact = 1.0` if any disputed variable is `robot_zone` or a goal box, else `0.4`.
- `doubt >= theta` → **revise**; else → **adopt**.

Rationale per ingredient: `(1 - disagreement)` scales everything by the WM's self-confidence; `(1 - validity_prob)` is the strongest single tell (the classic hallucination is believing a failed pick succeeded); `impact` deprioritizes disputes over boxes nobody cares about. Constants (`theta`, `u_max`, impact weights) live in `configs/w3.toml`, not code.

**OracleChecker** (same module): the env is deterministic, so the true post-action state is computable before execution via `transition(true_state, action)`. The oracle triggers a revision exactly when the asserted belief differs from the truth — the maximum-accuracy / minimum-spend bound. It peeks at ground truth by construction and is labeled as such everywhere.

## Runner integration — `src/lucid/agent.py`

`run_episode(..., gate=None)`:

- `gate=None` with a checker → W2's always-revise behavior, byte-identical (the merged arms must not change).
- With a gate: on each disagreement, `gate.decide(...)` routes to revise (existing revision loop), adopt (`working_belief = wm_state`, plus a free history line `note: your believed state after step N was corrected to <wm_state json>`), or ignore (`working_belief = belief`).
- Log rows gain two columns: `gate_decision` (`"revise" | "adopt" | "ignore" | ""`) and `doubt` (float, -1.0 when no disagreement occurred).
- `summarize()` gains `adopt_rate` and `ignore_rate` (per step) alongside the existing metrics.

The oracle arm passes `OracleChecker` in place of the WM checker; the runner treats them identically (same `check` interface).

## Evaluation — `configs/w3.toml` and the frontier

Six arms, seed 11, 100 episodes each (degrading to 60 under the budget cap):

| arm | checker | gate |
|---|---|---|
| `ungated` | — | — |
| `always_check` | WM | none (always revise) |
| `adaptive_lo` | WM | θ = 0.2 |
| `adaptive_mid` | WM | θ = 0.45 |
| `adaptive_hi` | WM | θ = 0.7 |
| `oracle` | ground truth | always revise on true divergence |

`ungated` and `always_check` replay entirely from the W2 cache (zero cost, zero network) and anchor the chart.

**`lucid-frontier` CLI**: reads `data/agent/summary.json`, renders the frontier — x: extra LLM calls vs. ungated (%), y: hallucinated-state rate (%); the three adaptive points drawn as a curve, fixed points for ungated / always_check / oracle — and writes `docs/assets/frontier.png` (committed; embedded in the README).

## Success criteria

1. **Frontier quality:** no adaptive point is dominated by `always_check` (worse on both axes), and at least one adaptive point has **≥ 30% fewer extra calls than `always_check` at ≤ 2.5% hallucinated-state rate**.
2. **Oracle bound** measured and plotted.
3. **Budget:** total new API spend ≤ $14, measured from per-arm token deltas at Haiku 4.5 pricing; actual spend reported in the README.
4. Tests remain zero-network; CI green.

Honesty note: adopt-mode fixes tracking but not the already-asserted belief, so adaptive arms are expected to trade a little accuracy for large call savings relative to `always_check` — the claim is a better *frontier*, not strict dominance everywhere. If the adaptive curve is everywhere dominated, that result is reported as-is (with the gate's decision logs enabling the diagnosis).

## Testing

- `DoubtGate.decide` unit tests: WM-uncertain → ignore; confident + invalid + goal-relevant → revise; confident + irrelevant + valid-ish → adopt; θ monotonicity (higher θ never converts an adopt into a revise).
- `OracleChecker`: flags exactly true divergences on crafted states.
- Gated `run_episode` integration test with the existing perfect-WM fixture and a scripted lying planner: adopt path corrects `working_belief` and appends the note; ignore path leaves the agent alone.
- `lucid-frontier` smoke test: crafted summary.json → PNG exists.
- All monkeypatched, zero network.

## Out of scope

- Learned gate (logistic regression over W2 logs) — possible W3.5 ablation.
- Live monitor UI, `@grounded` wrapper (W4).
- DuckDB.
