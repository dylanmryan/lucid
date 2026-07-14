# Lucid

A "lie detector" for AI agents: a tiny trained world model watches an LLM plan
and flags the moment the agent's beliefs diverge from reality.

W1 (this milestone): deterministic warehouse environment, rollout generator,
and a factored MLP ensemble world model with calibrated uncertainty.

## Run

    uv sync
    uv run lucid-gen-rollouts
    uv run lucid-train-wm
    uv run pytest

Design docs: `docs/superpowers/specs/`.
