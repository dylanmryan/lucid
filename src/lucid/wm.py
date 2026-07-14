"""Factored MLP ensemble world model: training, prediction, calibration metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from lucid.core import (
    Action,
    EnvConfig,
    Prediction,
    State,
    decode_state,
    encode,
    encoding_dim,
    target_indices,
)


class TransitionModel(nn.Module):
    """Shared trunk + one softmax head per state variable + a validity head."""

    def __init__(self, caps: EnvConfig, hidden: int = 256):
        super().__init__()
        zc = len(caps.zones)
        self.caps = caps
        self.trunk = nn.Sequential(
            nn.Linear(encoding_dim(caps), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.validity = nn.Linear(hidden, 1)
        self.robot = nn.Linear(hidden, zc)
        self.box_heads = nn.ModuleList(nn.Linear(hidden, zc + 1) for _ in range(caps.n_boxes))

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.validity(h).squeeze(-1), self.robot(h), [head(h) for head in self.box_heads]


def loss_fn(
    model: TransitionModel, x: torch.Tensor, valid: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """BCE on validity + CE per variable; targets are class indices, -1 = inactive slot."""
    v_logit, robot_logits, box_logits = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(v_logit, valid)
    loss = loss + nn.functional.cross_entropy(robot_logits, targets[:, 0])
    for i, logits in enumerate(box_logits):
        t = targets[:, 1 + i]
        if (t >= 0).any():
            loss = loss + nn.functional.cross_entropy(logits, t, ignore_index=-1)
    return loss


def load_dataset(path: Path, caps: EnvConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = pq.read_table(path)
    states = [State.from_json(s) for s in t.column("state").to_pylist()]
    actions = [Action.from_json(a) for a in t.column("action").to_pylist()]
    nexts = [State.from_json(s) for s in t.column("next_state").to_pylist()]
    x = torch.from_numpy(np.stack([encode(s, a, caps) for s, a in zip(states, actions)]))
    valid = torch.tensor(t.column("valid").to_pylist(), dtype=torch.float32)
    y = torch.from_numpy(np.stack([target_indices(s, caps) for s in nexts]))
    return x, valid, y


def train_model(
    x: torch.Tensor,
    valid: torch.Tensor,
    y: torch.Tensor,
    caps: EnvConfig,
    seed: int,
    epochs: int = 30,
    lr: float = 1e-3,
    batch: int = 256,
) -> TransitionModel:
    """One ensemble member: distinct init seed + bootstrap-resampled data."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(x), (len(x),), generator=g)  # bootstrap resample
    x, valid, y = x[idx], valid[idx], y[idx]
    model = TransitionModel(caps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch):
            j = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model, x[j], valid[j], y[j])
            loss.backward()
            opt.step()
    model.eval()
    return model


def var_names(caps: EnvConfig) -> list[str]:
    return ["robot_zone"] + [f"box_{i}" for i in range(caps.n_boxes)]


class Ensemble:
    def __init__(self, models: list[TransitionModel], caps: EnvConfig):
        self.models = models
        self.caps = caps

    @torch.no_grad()
    def forward_batch(self, x: torch.Tensor):
        """Stacked member outputs: v (K,B), robot (K,B,Z), boxes list of (K,B,Z+1)."""
        vs, rs, bs = [], [], []
        for m in self.models:
            v, r, b = m(x)
            vs.append(torch.sigmoid(v))
            rs.append(r.softmax(-1))
            bs.append([bi.softmax(-1) for bi in b])
        boxes = [
            torch.stack([bs[k][i] for k in range(len(self.models))])
            for i in range(self.caps.n_boxes)
        ]
        return torch.stack(vs), torch.stack(rs), boxes

    @torch.no_grad()
    def predict(self, state: State, action: Action) -> Prediction:
        n_boxes = len(state.box_zones)
        x = torch.from_numpy(encode(state, action, self.caps)).unsqueeze(0)
        v, r, boxes = self.forward_batch(x)
        names = var_names(self.caps)
        active = names[: 1 + n_boxes]
        mean_probs = {n: p.mean(0)[0].numpy() for n, p in zip(names, [r] + boxes)}
        idx = np.array([int(mean_probs[n].argmax()) for n in names])
        point = decode_state(idx, n_boxes, self.caps)
        member_idx = torch.stack([r.argmax(-1)] + [b.argmax(-1) for b in boxes], -1)[:, 0, :]
        point_idx = torch.from_numpy(idx)
        per_var = {
            n: float((member_idx[:, k] != point_idx[k]).float().mean())
            for k, n in enumerate(active)
        }
        diff = (member_idx[:, : 1 + n_boxes] != point_idx[: 1 + n_boxes]).any(-1)
        ent = float(np.mean([-(mean_probs[n] * np.log(mean_probs[n] + 1e-9)).sum() for n in active]))
        return Prediction(
            validity_prob=float(v.mean()),
            point_next_state=point,
            disagreement=float(diff.float().mean()),
            per_var_disagreement=per_var,
            entropy=ent,
            next_state_dist={n: mean_probs[n] for n in active},
        )


def train_ensemble(
    x: torch.Tensor, valid: torch.Tensor, y: torch.Tensor, caps: EnvConfig, k: int = 5, **kw
) -> Ensemble:
    return Ensemble([train_model(x, valid, y, caps, seed=s, **kw) for s in range(k)], caps)
