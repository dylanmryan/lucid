import pyarrow.parquet as pq

from lucid.core import EnvConfig
from lucid.env import generate_rollouts, write_splits

CFG = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)

SCHEMA = ["episode_id", "step", "state", "action", "valid", "next_state", "policy_tag"]


def test_rollout_schema_mix_and_invalids():
    t = generate_rollouts(CFG, n_episodes=40, seed=1)
    assert t.column_names == SCHEMA
    assert set(t.column("policy_tag").to_pylist()) == {"random_valid", "random_any", "scripted"}
    assert False in t.column("valid").to_pylist()  # negative examples for the validity head


def test_rollouts_deterministic():
    a = generate_rollouts(CFG, n_episodes=10, seed=2)
    b = generate_rollouts(CFG, n_episodes=10, seed=2)
    assert a.equals(b)


def test_splits_disjoint_by_episode(tmp_path):
    t = generate_rollouts(CFG, n_episodes=40, seed=3)
    write_splits(t, tmp_path)
    eps = [
        set(pq.read_table(tmp_path / f"{n}.parquet").column("episode_id").to_pylist())
        for n in ("train", "val", "test")
    ]
    assert all(eps)  # every split non-empty
    assert not (eps[0] & eps[1]) and not (eps[0] & eps[2]) and not (eps[1] & eps[2])
