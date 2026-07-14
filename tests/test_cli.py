import sys

from lucid.cli import gen_rollouts_main

CFG_TOML = """
[env]
n_boxes = 2
n_shelves = 2
max_steps = 20

[novel_env]
n_boxes = 3
n_shelves = 3
max_steps = 20

[caps]
n_boxes = 4
n_shelves = 4
max_steps = 20

[rollouts]
n_episodes = 30
n_novel_episodes = 5
seed = 1
out_dir = "{out}"

[train]
k = 2
epochs = 2
lr = 1e-3
batch = 64
out_dir = "{out}/wm"
"""


def test_gen_rollouts_cli(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG_TOML.replace("{out}", str(tmp_path / "rollouts")))
    monkeypatch.setattr(sys, "argv", ["lucid-gen-rollouts", "--config", str(cfg)])
    gen_rollouts_main()
    for f in ("train.parquet", "val.parquet", "test.parquet", "novel.parquet"):
        assert (tmp_path / "rollouts" / f).exists()
