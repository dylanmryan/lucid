"""Deterministic warehouse world: transitions, tasks, greedy solver, rollout generation."""

from __future__ import annotations

from lucid.core import HAND, Action, EnvConfig, State


def is_valid(state: State, action: Action, cfg: EnvConfig) -> bool:
    if action.kind == "move":
        return action.target in cfg.zones and action.target != state.robot_zone
    if action.kind == "pick":
        return (
            state.held is None
            and isinstance(action.target, int)
            and 0 <= action.target < cfg.n_boxes
            and state.box_zones[action.target] == state.robot_zone
        )
    return state.held is not None  # place


def transition(state: State, action: Action, cfg: EnvConfig) -> tuple[State, bool]:
    """Returns (next_state, valid). Invalid actions are no-ops."""
    if not is_valid(state, action, cfg):
        return state, False
    if action.kind == "move":
        return State(action.target, state.box_zones), True
    boxes = list(state.box_zones)
    if action.kind == "pick":
        boxes[action.target] = HAND
    else:  # place
        boxes[state.held] = state.robot_zone
    return State(state.robot_zone, tuple(boxes)), True


def all_actions(cfg: EnvConfig) -> list[Action]:
    return (
        [Action("move", z) for z in cfg.zones]
        + [Action("pick", i) for i in range(cfg.n_boxes)]
        + [Action("place")]
    )


def valid_actions(state: State, cfg: EnvConfig) -> list[Action]:
    return [a for a in all_actions(cfg) if is_valid(state, a, cfg)]
