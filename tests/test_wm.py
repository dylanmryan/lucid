import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from lucid.core import Action, EnvConfig, State
from lucid.env import generate_rollouts
from lucid.wm import (
    Ensemble,
    TransitionModel,
    disagreement_scores,
    ece,
    entropy_scores,
    evaluate,
    auroc,
    load_dataset,
    load_ensemble,
    loss_fn,
    save_ensemble,
    train_model,
)

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


def test_auroc_known_value():
    # classic example: 3 of 4 pos/neg pairs correctly ranked
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75


def test_auroc_handles_ties():
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_ece_well_calibrated_bucket():
    conf = np.array([0.8] * 10)
    correct = np.array([1.0] * 8 + [0.0] * 2)
    assert ece(conf, correct) < 0.011


def test_evaluate_and_novelty_scores(tmp_path):
    x, valid, y = _tiny_data(tmp_path)
    ens = _tiny_ensemble(tmp_path)
    m = evaluate(ens, x, valid, y)
    assert set(m) == {"validity_auroc", "next_state_exact_match", "ece"}
    assert all(0.0 <= v <= 1.0 for v in m.values())
    d = disagreement_scores(ens, x, y)
    e = entropy_scores(ens, x, y)
    assert d.shape == e.shape == (len(x),)


def test_save_load_round_trip(tmp_path):
    ens = _tiny_ensemble(tmp_path)
    save_ensemble(ens, tmp_path / "wm.pt")
    ens2 = load_ensemble(tmp_path / "wm.pt")
    s, a = State("receiving", ("receiving", "shelf_a")), Action("pick", 0)
    p1, p2 = ens.predict(s, a), ens2.predict(s, a)
    assert p1.point_next_state == p2.point_next_state
    assert p1.validity_prob == pytest.approx(p2.validity_prob)
