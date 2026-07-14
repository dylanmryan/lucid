"""Deterministic warehouse world: transitions, tasks, greedy solver, rollout generation."""

from __future__ import annotations

import random

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


def make_task(cfg: EnvConfig, rng: random.Random) -> tuple[State, dict[int, str]]:
    """Random initial state + goals: selected boxes must reach specific shelves."""
    state = State(
        rng.choice(cfg.zones),
        tuple(rng.choice(cfg.zones) for _ in range(cfg.n_boxes)),
    )
    shelves = [z for z in cfg.zones if z.startswith("shelf_")]
    goal_boxes = rng.sample(range(cfg.n_boxes), rng.randint(1, cfg.n_boxes))
    goals = {i: rng.choice(shelves) for i in goal_boxes}
    return state, goals


def goals_met(state: State, goals: dict[int, str]) -> bool:
    return all(state.box_zones[i] == z for i, z in goals.items())


def solve(state: State, goals: dict[int, str], cfg: EnvConfig) -> list[Action]:
    """Greedy plan: fetch and shelve each unmet goal box in index order. Near-optimal is enough."""
    plan: list[Action] = []
    cur = state
    if cur.held is not None:
        plan.append(Action("place"))
        cur, _ = transition(cur, plan[-1], cfg)
    for box, dest in sorted(goals.items()):
        if cur.box_zones[box] == dest:
            continue
        if cur.robot_zone != cur.box_zones[box]:
            plan.append(Action("move", cur.box_zones[box]))
            cur, _ = transition(cur, plan[-1], cfg)
        plan.append(Action("pick", box))
        cur, _ = transition(cur, plan[-1], cfg)
        if cur.robot_zone != dest:
            plan.append(Action("move", dest))
            cur, _ = transition(cur, plan[-1], cfg)
        plan.append(Action("place"))
        cur, _ = transition(cur, plan[-1], cfg)
    return plan


class WarehouseEnv:
    """Thin stateful wrapper for episode stepping."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def reset(self, seed: int) -> tuple[State, dict[int, str]]:
        self.state, self.goals = make_task(self.cfg, random.Random(seed))
        self.steps = 0
        return self.state, self.goals

    def step(self, action: Action) -> tuple[State, bool, bool]:
        """Returns (state, valid, done)."""
        self.state, valid = transition(self.state, action, self.cfg)
        self.steps += 1
        done = goals_met(self.state, self.goals) or self.steps >= self.cfg.max_steps
        return self.state, valid, done
