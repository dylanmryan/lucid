"""Shared types: state, actions, env config, world-model prediction, encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

HAND = "hand"  # sentinel zone: box is in the robot's gripper

FIXED_ZONES = ("receiving", "staging", "packing")


@dataclass(frozen=True)
class EnvConfig:
    n_boxes: int = 4
    n_shelves: int = 3
    max_steps: int = 40

    @property
    def zones(self) -> tuple[str, ...]:
        shelves = tuple(f"shelf_{chr(ord('a') + i)}" for i in range(self.n_shelves))
        return FIXED_ZONES + shelves

    @property
    def boxes(self) -> tuple[str, ...]:
        return tuple(f"box_{i}" for i in range(self.n_boxes))


@dataclass(frozen=True)
class State:
    robot_zone: str
    box_zones: tuple[str, ...]  # index i = box_i's zone; HAND means held by the robot

    @property
    def held(self) -> int | None:
        return self.box_zones.index(HAND) if HAND in self.box_zones else None

    def to_json(self) -> str:
        return json.dumps({"robot_zone": self.robot_zone, "box_zones": list(self.box_zones)})

    @classmethod
    def from_json(cls, s: str) -> State:
        d = json.loads(s)
        return cls(d["robot_zone"], tuple(d["box_zones"]))


@dataclass(frozen=True)
class Action:
    kind: Literal["move", "pick", "place"]
    target: str | int | None = None  # zone name for move, box index for pick, None for place

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "target": self.target})

    @classmethod
    def from_json(cls, s: str) -> Action:
        d = json.loads(s)
        return cls(d["kind"], d["target"])


@dataclass
class Prediction:
    """World-model output — the currency of the gate, monitor, and eval."""

    validity_prob: float
    point_next_state: State
    disagreement: float  # fraction of members whose decoded state differs from the point prediction
    per_var_disagreement: dict[str, float]  # keys: "robot_zone", "box_0", ...
    entropy: float  # mean predictive entropy over active variables (nats)
    next_state_dist: dict[str, np.ndarray]  # ensemble-mean probs per active variable


def zone_index(zone: str, caps: EnvConfig) -> int:
    return caps.zones.index(zone)


def encoding_dim(caps: EnvConfig) -> int:
    zc = len(caps.zones)
    return zc + caps.n_boxes * (zc + 1) + 3 + zc + caps.n_boxes


def encode(state: State, action: Action, caps: EnvConfig) -> np.ndarray:
    """One-hot (state, action) into a fixed vector sized by caps; inactive box slots stay zero."""
    if len(state.box_zones) > caps.n_boxes:
        raise ValueError(f"state has {len(state.box_zones)} boxes; caps allow {caps.n_boxes}")
    zc = len(caps.zones)
    robot = np.zeros(zc, np.float32)
    robot[zone_index(state.robot_zone, caps)] = 1.0
    parts = [robot]
    for i in range(caps.n_boxes):
        slot = np.zeros(zc + 1, np.float32)
        if i < len(state.box_zones):
            z = state.box_zones[i]
            slot[zc if z == HAND else zone_index(z, caps)] = 1.0
        parts.append(slot)
    kind = np.zeros(3, np.float32)
    kind[("move", "pick", "place").index(action.kind)] = 1.0
    parts.append(kind)
    zone_t = np.zeros(zc, np.float32)
    if action.kind == "move":
        zone_t[zone_index(action.target, caps)] = 1.0
    parts.append(zone_t)
    box_t = np.zeros(caps.n_boxes, np.float32)
    if action.kind == "pick":
        box_t[action.target] = 1.0
    parts.append(box_t)
    return np.concatenate(parts)


def target_indices(next_state: State, caps: EnvConfig) -> np.ndarray:
    """Per-variable class targets: [robot_zone, box_0, ..., box_{caps.n_boxes-1}]; -1 = inactive."""
    zc = len(caps.zones)
    out = np.full(1 + caps.n_boxes, -1, np.int64)
    out[0] = zone_index(next_state.robot_zone, caps)
    for i, z in enumerate(next_state.box_zones):
        out[1 + i] = zc if z == HAND else zone_index(z, caps)
    return out


def decode_state(indices: np.ndarray, n_boxes: int, caps: EnvConfig) -> State:
    """Inverse of target_indices over the first 1 + n_boxes (active) entries."""
    zc = len(caps.zones)
    if not 0 <= indices[0] < zc or not all(0 <= indices[1 + i] <= zc for i in range(n_boxes)):
        raise ValueError(f"indices out of range for n_boxes={n_boxes}: {indices[: 1 + n_boxes]}")
    robot = caps.zones[indices[0]]
    boxes = tuple(
        HAND if indices[1 + i] == zc else caps.zones[indices[1 + i]] for i in range(n_boxes)
    )
    return State(robot, boxes)
