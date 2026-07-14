import pyarrow.parquet as pq
import torch

from lucid.core import EnvConfig
from lucid.env import generate_rollouts
from lucid.wm import TransitionModel, load_dataset, loss_fn, train_model

CFG = EnvConfig(n_boxes=2, n_shelves=2, max_steps=30)
CAPS = EnvConfig(n_boxes=3, n_shelves=3)


def _tiny_data(tmp_path):
    t = generate_rollouts(CFG, n_episodes=40, seed=5)
    p = tmp_path / "t.parquet"
    pq.write_table(t, p)
    return load_dataset(p, CAPS)


def test_load_dataset_shapes(tmp_path):
    x, valid, y = _tiny_data(tmp_path)
    assert x.dtype == torch.float32 and x.ndim == 2
    assert valid.shape == (len(x),) and y.shape == (len(x), 1 + CAPS.n_boxes)
    assert (y[:, 3] == -1).all()  # box_2 slot inactive for a 2-box config


def test_training_learns(tmp_path):
    x, valid, y = _tiny_data(tmp_path)
    torch.manual_seed(0)
    before = loss_fn(TransitionModel(CAPS), x, valid, y).item()
    model = train_model(x, valid, y, CAPS, seed=0, epochs=20)
    after = loss_fn(model, x, valid, y).item()
    assert after < before / 2
