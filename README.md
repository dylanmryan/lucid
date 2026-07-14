# Lucid

A "lie detector" for AI agents: a tiny trained world model watches an LLM plan
and flags the moment the agent's beliefs diverge from reality.

W1 (this milestone): deterministic warehouse environment, rollout generator,
and a factored MLP ensemble world model with calibrated uncertainty.

## Run

    uv sync
    uv run lucid-gen-rollouts
    uv run lucid-train-wm
    uv run lucid-run-agent    # needs ANTHROPIC_API_KEY on first run; cached afterwards
    uv run pytest

Design docs: `docs/superpowers/specs/`.

## W1 results

World model: 5-member factored MLP ensemble trained on 69k transitions from the
deterministic warehouse env (config: `configs/w1.toml`; test split 8,721
transitions, novel-config probe 9,593).

| metric | value |
|---|---|
| validity AUROC (test) | 1.000 |
| next-state exact match (test) | 1.000 |
| ECE (per-variable confidence) | 1.9e-05 |
| novelty AUROC — disagreement | 1.000 |
| novelty AUROC — entropy | 1.000 |

The uncertainty signal separates novel-config states from familiar ones with no
overlap (in-distribution disagreement max 0.000 vs. novel minimum 0.600) — the
signal the doubt-budget gate spends in W3. Plots: `data/wm/reliability.png`,
`data/wm/separation.png` (regenerate with the two CLI commands above).
