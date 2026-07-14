# Lucid — W2 Design: LLM Planner, Hallucination Metric, Check→Revise Loop

**Date:** 2026-07-14
**Status:** Approved
**Scope:** The W2 slice: LiteLLM planner with disk-cached calls, open-loop belief tracking, hallucinated-state metric, and the draft→check→revise loop wired against the W1 world model. Two arms (`ungated`, `always_check`). The adaptive doubt-budget gate and four-arm frontier are W3; the live monitor and `@grounded` wrapper are W4.

## Context

W1 (merged) delivered the deterministic warehouse env, rollout generator, and a 5-member factored MLP ensemble world model with perfect in-distribution accuracy and total novelty separation (see README W1 results). W2 adds the LLM agent whose hallucinations the world model exists to catch.

Decisions made during brainstorming:

- **Planner LLM:** Claude Haiku 4.5 via LiteLLM (`claude-haiku-4-5`), temperature 0. Cheap enough for hundreds of episodes, weak enough to hallucinate on long-horizon belief tracking.
- **Observability:** valid-flag only. The agent sees the initial state + goals, then after each action only whether it was valid — never the true state (GILP's open-loop setting). Belief drift is what makes hallucination measurable and gating meaningful.
- **Interaction shape:** stepwise JSON protocol (one call per step, asserted belief each step) — not plan-then-execute, not freeform ReAct.

## Architecture

One new module `src/lucid/agent.py` (+ a `lucid-run-agent` CLI entry in `cli.py`, `configs/w2.toml`, and `litellm` in pyproject). Four units inside `agent.py`:

- **`CachedLLM`** — LiteLLM wrapper. Cache key = SHA-256 of (model, messages, temperature); value = raw completion text at `data/llm_cache/<hash>.json` (gitignored). Tracks hit/miss counts for the summary. Cache hit → zero network.
- **`Planner`** — owns the prompt protocol. `plan_step(...) -> (Action, State, meta)`: builds messages, calls `CachedLLM`, parses strict JSON, retries malformed output ≤ 2 times with the parse error quoted back.
- **`Checker`** — wraps the W1 `Ensemble`. Input: previous *believed* state + chosen action (never true state). Diffs the WM's `point_next_state` against the planner's asserted belief per variable; on mismatch emits a revision note (e.g., "world model predicts box_2 still at receiving, validity 0.03 — your pick likely failed"). Planner re-drafts, ≤ 2 revisions per step; after that the WM's prediction becomes the working belief and the episode continues.
- **Episode runner** — glues planner (+ optional checker) to `WarehouseEnv`. Per step: history → Planner → (action, belief) → [Checker → maybe revise] → `env.step(action)` → valid flag appended to history → log row.

**Per-step log row** (parquet, one table per arm): episode_id, step, true_state, believed_state, wm_predicted_state, action, valid, n_revisions, n_parse_retries, parse_failure, cache_hit, tokens_in, tokens_out. The metric never influences the run; it is computed offline from logs. This log is already shaped like what the W4 monitor streams.

W2 runs use the same env config the WM was trained on (4 boxes / 3 shelves) so checker accuracy is in-distribution by construction.

## Planner protocol

**System prompt** (static; generated from the actual `EnvConfig` so prompt and env cannot drift): warehouse rules (zones, one box in hand, three actions with preconditions), the JSON reply format, and the belief-tracking obligation ("after each action you will only be told whether it was valid — track the state yourself").

**Per-step user message:** goals, initial state (`State.to_json` format), full history as compact lines — `step 3: {"kind": "move", "target": "shelf_a"} -> valid` — plus the checker's disagreement note on revise turns.

**Required reply:**

```json
{"action": {"kind": "pick", "target": 2}, "believed_next_state": {"robot_zone": "shelf_a", "box_zones": ["shelf_a", "hand", "packing", "staging"]}}
```

**Parsing** is strict and defensive so malformed output cannot contaminate the metric: extract the first `{...}` block, `json.loads`, validate action kind/target types, and require `believed_next_state` to decode to a well-formed `State` with the correct box count (same validation posture as W1's `decode_state` guards). Failure → one retry with the error quoted back; after 2 failed retries the step logs `parse_failure=True`, **no action is executed** (true state and working belief unchanged; history records the failure), and the episode continues. Parse failures are excluded from the hallucination numerator and denominator and tracked as their own rate.

**Determinism:** temperature 0, fixed model string, `ANTHROPIC_API_KEY` from the environment. Identical config + warm cache → byte-identical, free rerun.

## Metrics and arms

**Hallucinated-state rate** (headline): over all non-parse-failure steps, the fraction where the asserted `believed_next_state` differs from the env's true post-action state on any variable. Reported per arm. Compounding drift counts (as in GILP); `first_divergence_step` per episode is also reported to make compounding visible.

**Secondary metrics:** task success rate, mean steps, LLM calls/episode, revisions/step, invalid-action rate, cache hit rate, cost estimate from LiteLLM token counts.

**Arms:**
- `ungated` — planner alone; the baseline number (paper analog: 0.176).
- `always_check` — checker every step; proves the revise loop (paper analog: 0.035) at ~2× calls. W3's adaptive gate attacks that cost.

## Success criteria (measured, in `data/agent/summary.json`)

1. Ungated baseline hallucinated-state rate measured over ≥ 100 episodes and **≥ 5%** — the task genuinely induces hallucination. Contingency if Haiku is too accurate: raise `max_steps` and/or `n_boxes` in `configs/w2.toml` and retrain the WM at the bigger config (one command), then re-measure. The knob is documented; the guess is not hard-coded.
2. `always_check` cuts the hallucinated-state rate **≥ 50% relative** vs. ungated.
3. Parse-failure rate **< 2%** of calls.
4. Rerun with warm cache performs **zero** network calls (asserted via cache-stats counters in the summary).

## Config and CLI

`configs/w2.toml`:

```toml
[agent]
model = "claude-haiku-4-5"        # LiteLLM model string
max_parse_retries = 2
max_revisions = 2
cache_dir = "data/llm_cache"

[run]
env = { n_boxes = 4, n_shelves = 3, max_steps = 40 }   # = WM training config
n_episodes = 100
seed = 11
arms = ["ungated", "always_check"]
wm_checkpoint = "data/wm/ensemble.pt"
out_dir = "data/agent"
```

`lucid-run-agent --config configs/w2.toml [--arm ungated]`: runs requested arms sequentially, writes `data/agent/<arm>.parquet` + `data/agent/summary.json`, prints the summary. Idempotent under the cache.

## Testing and CI

Unit tests monkeypatch `litellm.completion` (zero network in CI):

- parser: accepts good JSON; malformed → retry → `parse_failure` path,
- prompt builder: includes valid-flag history and revision notes,
- `CachedLLM`: writes then hits its cache (tmp dir), counters correct,
- checker: emits revision note on WM/belief mismatch; adopts WM belief after bounded revisions,
- end-to-end episode with a scripted fake LLM asserting the full log schema.

The real-API path is exercised only by the local CLI run. CI workflow unchanged; litellm must not require a key at import time (it doesn't).

## Out of scope for W2

- Adaptive doubt-budget gate, four-arm eval, cost/accuracy frontier chart (W3)
- DuckDB (parquet + summary dict suffice until the frontier queries in W3)
- Live monitor UI, `@grounded` wrapper (W4)
