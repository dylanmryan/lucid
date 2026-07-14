import pyarrow.parquet as pq
import torch

from lucid.core import Action, EnvConfig, State
from lucid.env import generate_rollouts
from lucid.wm import Ensemble, TransitionModel, load_dataset, loss_fn, train_model

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


def _tiny_ensemble(tmp_path, k=2):
    x, valid, y = _tiny_data(tmp_path)
    return Ensemble([train_model(x, valid, y, CAPS, seed=s, epochs=5) for s in range(k)], CAPS)


def test_predict_fields(tmp_path):
    ens = _tiny_ensemble(tmp_path)
    p = ens.predict(State("receiving", ("receiving", "shelf_a")), Action("pick", 0))
    assert 0.0 <= p.validity_prob <= 1.0
    assert isinstance(p.point_next_state, State) and len(p.point_next_state.box_zones) == 2
    assert set(p.per_var_disagreement) == {"robot_zone", "box_0", "box_1"}
    assert 0.0 <= p.disagreement <= 1.0 and p.entropy >= 0.0
    assert p.next_state_dist["box_0"].shape == (len(CAPS.zones) + 1,)
