# Lucid

A "lie detector" for AI agents: a tiny trained world model watches an LLM plan
and flags the moment the agent's beliefs diverge from reality.

W1: deterministic warehouse environment, rollout generator, and a factored MLP
ensemble world model with calibrated uncertainty. W2: an LLM planner tracking
its own beliefs open-loop, the hallucinated-state metric, and the
draft→check→revise loop that lets the world model catch the lies.

## Run

    uv sync
    uv run lucid-gen-rollouts
    uv run lucid-train-wm
    uv run lucid-run-agent    # needs ANTHROPIC_API_KEY on first run; cached afterwards
    uv run pytest

Design docs: `docs/superpowers/specs/`.

## W2 results — hallucination and the check→revise loop

Claude Haiku 4.5 plans in an 8-box warehouse, seeing only its goals, the
initial state, and per-action valid flags — never the true state. Each step it
asserts a believed next state; **hallucinated-state rate** = fraction of steps
where that belief is wrong. The `always_check` arm diffs every belief against
the W1 world model (which never sees the true state either) and asks the
planner to revise on disagreement, at most twice per step.

100 episodes per arm (`configs/w2.toml`, seed 11):

| metric | ungated | always_check |
|---|---|---|
| hallucinated-state rate | **13.6%** | **1.2%** (−91% relative) |
| task success | 99% | 100% |
| mean first divergence | step 7.1 | step 11.3 |
| LLM calls / episode | 17.62 | 19.81 (+12.4%) |
| invalid-action rate | 1.8% | 0.6% |
| parse failures | 0 / 1762 calls | 0 / 1981 calls |

Reference points from the GILP paper (arXiv:2606.27806): 17.6% → 3.5% for
~22% extra calls. Lucid's checker cuts deeper (−91% vs. −80%) for roughly half
the call overhead, because revisions only fire on the ~14% of steps where the
world model actually disagrees — the observation the W3 adaptive doubt-budget
gate is built on.

Every LLM call is content-hash-cached to `data/llm_cache/`; a warm rerun of
the full eval makes zero network calls and reproduces these numbers
byte-for-byte.

## W1 results — the world model

5-member factored MLP ensemble trained on 156k transitions from the
deterministic warehouse env (config: `configs/w1.toml`, 8 boxes / 5 shelves;
held-out test 18.9k transitions, novel-config probe 16.5k). The env was bumped
from 4 to 8 boxes per the W2 spec's contingency — Haiku barely hallucinated on
4-box tasks.

| metric | value |
|---|---|
| validity AUROC (test) | 0.996 |
| next-state exact match (test) | 0.9998 |
| ECE (per-variable confidence) | 9.7e-04 |
| novelty AUROC — disagreement | 1.000 |
| novelty AUROC — entropy | 1.000 |

Ensemble disagreement and entropy separate never-seen configurations from
familiar ones essentially perfectly — the uncertainty signal the doubt-budget
gate spends in W3. Plots: `data/wm/reliability.png`, `data/wm/separation.png`
(regenerate with the CLI commands above).

## Roadmap

- **W3** — adaptive doubt-budget gate: spend checks where disagreement and
  uncertainty are high; four-arm eval (ungated / fixed / adaptive / oracle) and
  the cost-accuracy frontier chart.
- **W4** — live grounding monitor (belief vs. reality streaming over
  websockets) and the `@grounded` wrapper for arbitrary tool-using agents.
