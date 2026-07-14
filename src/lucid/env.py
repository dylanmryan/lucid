"""Deterministic warehouse world: transitions, tasks, greedy solver, rollout generation."""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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


POLICY_MIX = (("random_valid", 0.4), ("random_any", 0.3), ("scripted", 0.3))


def _episode(env: WarehouseEnv, seed: int, policy: str, rng: random.Random) -> list[tuple]:
    rows = []
    state, goals = env.reset(seed)
    script = solve(state, goals, env.cfg) if policy == "scripted" else None
    for step in range(env.cfg.max_steps):
        if policy == "scripted":
            if step >= len(script):
                break
            action = script[step]
        elif policy == "random_valid":
            action = rng.choice(valid_actions(state, env.cfg))
        else:  # random_any: includes invalid actions on purpose
            action = rng.choice(all_actions(env.cfg))
        next_state, valid, done = env.step(action)
        rows.append(
            (seed, step, state.to_json(), action.to_json(), valid, next_state.to_json(), policy)
        )
        state = next_state
        if done:
            break
    return rows


def generate_rollouts(
    cfg: EnvConfig, n_episodes: int, seed: int, policy_mix=POLICY_MIX
) -> pa.Table:
    rng = random.Random(seed)
    env = WarehouseEnv(cfg)
    policies = [p for p, _ in policy_mix]
    weights = [w for _, w in policy_mix]
    rows = []
    for ep in range(n_episodes):
        policy = rng.choices(policies, weights)[0]
        rows.extend(_episode(env, seed * 100_000 + ep, policy, rng))
    names = ["episode_id", "step", "state", "action", "valid", "next_state", "policy_tag"]
    return pa.table(dict(zip(names, map(list, zip(*rows)))))


def write_splits(table: pa.Table, out_dir: Path) -> None:
    """Split by episode (never by transition): last digit of episode id buckets 10 ways."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket = [e % 10 for e in table.column("episode_id").to_pylist()]
    splits = {"test": lambda b: b == 0, "val": lambda b: b == 1, "train": lambda b: b >= 2}
    for name, keep in splits.items():
        pq.write_table(table.filter(pa.array([keep(b) for b in bucket])), out_dir / f"{name}.parquet")
