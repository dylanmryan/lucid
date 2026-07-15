"""CLI entry points: lucid-gen-rollouts, lucid-train-wm."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lucid.core import EnvConfig
from lucid.env import generate_rollouts, write_splits
from lucid.wm import (
    auroc,
    disagreement_scores,
    entropy_scores,
    evaluate,
    load_dataset,
    per_variable_confidence,
    reliability_plot,
    save_ensemble,
    separation_plot,
    train_ensemble,
)


def _cfg_and_arm(default: str) -> tuple[dict, str | None]:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default)
    p.add_argument("--arm", default=None)
    args = p.parse_args()
    with open(args.config, "rb") as f:
        return tomllib.load(f), args.arm


def gen_rollouts_main() -> None:
    cfg, _ = _cfg_and_arm("configs/w1.toml")
    ro = cfg["rollouts"]
    out = Path(ro["out_dir"])
    table = generate_rollouts(EnvConfig(**cfg["env"]), ro["n_episodes"], ro["seed"])
    write_splits(table, out)
    novel = generate_rollouts(EnvConfig(**cfg["novel_env"]), ro["n_novel_episodes"], ro["seed"] + 1)
    pq.write_table(novel, out / "novel.parquet")
    print(f"{table.num_rows} transitions -> {out}, {novel.num_rows} novel-config transitions")


def train_wm_main() -> None:
    cfg, _ = _cfg_and_arm("configs/w1.toml")
    caps = EnvConfig(**cfg["caps"])
    tr = cfg["train"]
    data = Path(cfg["rollouts"]["out_dir"])
    out = Path(tr["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    x, valid, y = load_dataset(data / "train.parquet", caps)
    ens = train_ensemble(
        x, valid, y, caps, k=tr["k"], epochs=tr["epochs"], lr=tr["lr"], batch=tr["batch"]
    )
    save_ensemble(ens, out / "ensemble.pt")

    xt, vt, yt = load_dataset(data / "test.parquet", caps)
    xn, _, yn = load_dataset(data / "novel.parquet", caps)
    metrics = evaluate(ens, xt, vt, yt)
    d_id, d_nov = disagreement_scores(ens, xt, yt), disagreement_scores(ens, xn, yn)
    e_id, e_nov = entropy_scores(ens, xt, yt), entropy_scores(ens, xn, yn)
    labels = np.concatenate([np.zeros(len(d_id)), np.ones(len(d_nov))])
    metrics["novelty_auroc_disagreement"] = auroc(labels, np.concatenate([d_id, d_nov]))
    metrics["novelty_auroc_entropy"] = auroc(labels, np.concatenate([e_id, e_nov]))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    conf, correct = per_variable_confidence(ens, xt, yt)
    reliability_plot(conf, correct, out / "reliability.png")
    separation_plot(d_id, d_nov, out / "separation.png")
    print(json.dumps(metrics, indent=2))


def run_agent_main() -> None:
    from lucid.agent import run_arm

    cfg, arm = _cfg_and_arm("configs/w2.toml")
    arms = [arm] if arm else cfg["run"]["arms"]
    bad = [a for a in arms if a not in cfg["run"]["arms"]]
    if bad:
        raise SystemExit(f"--arm must be one of {cfg['run']['arms']}, got {bad}")
    summaries = {a: run_arm(a, cfg["run"], cfg["agent"], cfg.get("gate")) for a in arms}
    out = Path(cfg["run"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    existing.update(summaries)
    summary_path.write_text(json.dumps(existing, indent=2))
    print(json.dumps(summaries, indent=2))
