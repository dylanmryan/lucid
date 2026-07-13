# Lucid — W1 Design: Warehouse Environment + World Model

**Date:** 2026-07-13
**Status:** Approved
**Scope:** High-level architecture for the whole system; detailed spec for the W1 slice (environment + rollout generator + world model with uncertainty head). W2–W4 get their own spec cycles.

## Context

Lucid implements the GILP idea ([arXiv:2606.27806](https://arxiv.org/abs/2606.27806)): pair an LLM planner with a small trained world model and a consistency gate; on disagreement, the agent revises. Lucid adds three things: a real-time grounding monitor (belief vs. reality, live), an adaptive "doubt budget" gate (spend checks where the model is most likely wrong), and a universal `@grounded` wrapper.

Decisions made during brainstorming:

- **Scope:** architecture now, detailed W1 spec, build W1 first.
- **Environment:** warehouse world (zone-based logistics), not web-shop.
- **World model:** factored MLP ensemble (approach A) behind a protocol so other models can swap in later.
- **Naming:** repo and package are `lucid`.
- **Tooling:** uv + ruff + pytest + GitHub Actions.

## Architecture

One package, `lucid`, six subpackages. The load-bearing piece is the shared data contract in `core/`.

- **`lucid/core/`** — shared types. `State` (frozen dataclass: box locations, robot zone, held item), `Action` (`Move`, `Pick`, `Place`), and `Prediction`: `{validity_prob, next_state_dist, point_next_state, disagreement, entropy}`. `Prediction` is the currency of the system: the gate spends budget on it, the monitor renders it, the eval logs it.
- **`lucid/env/`** — `WarehouseEnv` (gymnasium-style, deterministic, explicit preconditions; invalid actions are no-ops returning `valid=False`) + rollout generator.
- **`lucid/wm/`** — `WorldModel` protocol (`predict(state, action) -> Prediction`); `FactoredMLPEnsemble` implementation.
- **`lucid/gate/`** (W3) — `DoubtBudgetGate`: consumes `Prediction`s + a budget; decides check vs. cruise.
- **`lucid/agent/`** (W2) — LiteLLM planner, draft→check→revise loop; all LLM calls content-hash-cached to disk.
- **`lucid/monitor/`** (W4) — FastAPI + websockets; streams belief vs. predicted vs. true state per step.
- **`lucid/eval/`** (W2–W3) — DuckDB episode logs, four-arm harness (LLM-only / fixed-gate / adaptive-gate / oracle), cost/accuracy frontier chart.

**Hallucination metric (fixed from day one):** at each step the agent asserts a belief state; hallucinated-state rate = fraction of steps where the asserted belief differs from the env's true state on any variable. The env is the ground-truth oracle; the world model is the deployable approximation. The gate never peeks at true state.

## W1: Warehouse environment

**World.** Zone-based, no grid. Zones: `receiving`, `staging`, K shelves (`shelf_a`, …), `packing`. One robot occupying a zone, holding at most one box. N boxes, each at a zone or in the robot's hand.

**State** = `(robot_zone, held_box | None, {box_id → zone})`. Frozen, hashable, JSON-serializable (the LLM planner and monitor read/write states as text).

**Actions and preconditions.**

| Action | Valid iff |
|---|---|
| `Move(zone)` | target ≠ current zone |
| `Pick(box)` | robot in box's zone, hands empty |
| `Place()` | holding a box (places at current zone) |

Invalid actions are no-ops returning `valid=False` — a hallucinating agent will emit them, and that event is exactly what Lucid catches.

**Tasks.** A task = a set of goal assignments (`box_3 → shelf_b`), generated from a seed. Success when all goals hold; episodes capped at a max step count. Env config parameterizes `n_boxes`, `n_shelves` — training on small configs and probing bigger ones tests whether the uncertainty head fires on novel states.

**Rollout generator.** Three mixed policies, ratios configurable:

1. random-valid (broad coverage),
2. random-including-invalid (negative examples for the validity head),
3. scripted near-optimal (coverage of the goal-directed region the planner inhabits).

Output: transitions table `(episode_id, step, state, action, valid, next_state, policy_tag)`, written as parquet, loaded via DuckDB. Splits by episode, not by transition, plus a held-out **novel-config split** (more boxes/shelves than training saw) reserved for calibration eval.

## W1: World model — `FactoredMLPEnsemble`

**Encoding.** State and action encode to a fixed vector sized by `MAX_BOXES`/`MAX_ZONES` caps (not the current config): one-hot robot zone, one-hot held box (+none), one-hot location per box slot (+"in hand"), inactive slots zeroed; action = one-hot type + one-hot parameters. Sizing to the cap lets a model trained on 4 boxes be asked about 8 — it'll be wrong, and its uncertainty should know it.

**Heads.** Shared MLP trunk (~2 hidden layers, a few hundred units), then per-variable heads:

- `validity` — sigmoid, BCE against the env's `valid` flag,
- `robot_zone'` — softmax over zones,
- `held'` — softmax over boxes + none,
- `box_i location'` — softmax over zones + hand, one head per box slot; inactive slots masked out of the loss.

**Ensemble.** K=5 members, different init seeds + bootstrap-resampled training data. Small enough that all five train in minutes on CPU/MPS.

**Prediction assembly.**

- `point_next_state`: per-variable argmax of ensemble-mean distributions, decoded to a `State`,
- `disagreement`: fraction of members whose decoded next state differs from the majority, plus a per-variable version (feeds the monitor's "unsure *where box_3 is*" display),
- `entropy`: mean predictive entropy of ensemble-averaged distributions,
- `validity_prob`: ensemble-mean validity.

**Success criteria** (measured, written to a metrics JSON + plots):

1. In-distribution test: validity AUROC and next-state exact-match accuracy ≳99% (deterministic small env; less means a bug).
2. Calibration: ECE + reliability diagram for next-state confidence.
3. **Uncertainty separates novel from familiar:** disagreement/entropy as an in-distribution vs. novel-config classifier achieves AUROC ≥ 0.8. If this fails the doubt budget has nothing to spend on — W1 isn't done until it passes.

**Deliverables:** `uv run lucid-gen-rollouts`, `uv run lucid-train-wm`, checkpoints, metrics JSON, reliability + separation plots.

## Repo layout, testing, CI

```
lucid/
├── pyproject.toml          # uv; deps: torch, gymnasium, duckdb, pyarrow, matplotlib
├── src/lucid/
│   ├── core/               # State, Action, Prediction, encoders
│   ├── env/                # WarehouseEnv, task generator, rollout generator
│   ├── wm/                 # WorldModel protocol, FactoredMLPEnsemble, training, calibration
│   ├── gate/  agent/  monitor/  eval/    # stubs now, filled in W2–W4
│   └── cli.py              # lucid-gen-rollouts, lucid-train-wm entry points
├── tests/
├── data/                   # gitignored: parquet rollouts, checkpoints, metrics
└── docs/superpowers/specs/
```

**Config:** plain dataclasses + a TOML file for experiment settings (env size, rollout mix, training hyperparams). No hydra/omegaconf.

**Testing (CI-fast, CPU-only, zero network, <1 min):** env invariants (determinism given seed, precondition table, no-op on invalid), encoder round-trips (encode→decode == identity), rollout schema, one smoke test training a 1-member ensemble on ~100 transitions asserting loss decreases. Real training runs happen locally via the CLIs.

**CI:** one GitHub Actions workflow: `uv sync`, `ruff check`, `ruff format --check`, `pytest`.

**Not dependencies yet:** FastAPI, LiteLLM, websockets — they arrive with their weekends (W2/W4).

## Out of scope for W1

- LLM planner, draft→check→revise loop, hallucination measurement runs (W2)
- Doubt-budget gate + four-arm eval + frontier chart (W3)
- Live monitor UI, `@grounded` wrapper (W4)
