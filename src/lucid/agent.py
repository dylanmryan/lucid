"""LLM planner: cached LiteLLM calls, belief tracking, draft->check->revise loop."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import litellm

from lucid.core import HAND, Action, EnvConfig, State


@dataclass(frozen=True)
class Task:
    initial: State
    goals: dict[int, str]


def system_prompt(cfg: EnvConfig) -> str:
    zones = ", ".join(cfg.zones)
    return (
        f"You control a warehouse robot. Zones: {zones}. "
        f"Boxes are box_0..box_{cfg.n_boxes - 1}. The robot holds at most one box.\n"
        "Actions and preconditions:\n"
        '- {"kind": "move", "target": "<zone>"} — valid unless already in that zone.\n'
        '- {"kind": "pick", "target": <box index>} — valid iff the robot is in that box\'s zone '
        "and holds nothing.\n"
        '- {"kind": "place", "target": null} — puts the held box in the current zone; '
        "valid iff holding a box.\n"
        "Invalid actions do nothing.\n"
        "After each action you are told ONLY whether it was valid — never the state. "
        "Track the state yourself.\n"
        "Reply with EXACTLY one JSON object and nothing else:\n"
        '{"action": {"kind": "...", "target": ...}, '
        '"believed_next_state": {"robot_zone": "<zone>", "box_zones": ["<zone or hand>", ...]}}\n'
        f'"box_zones" lists all {cfg.n_boxes} boxes in index order; "hand" means held.'
    )


def user_message(task: Task, history: list[str], note: str | None) -> str:
    goals = ", ".join(f"box_{i} -> {z}" for i, z in sorted(task.goals.items()))
    lines = [f"Goals: {goals}", f"Initial state: {task.initial.to_json()}"]
    if history:
        lines.append("History:")
        lines.extend(history)
    if note:
        lines.append(f"Checker note: {note}")
    lines.append("Next action?")
    return "\n".join(lines)


def parse_reply(text: str, cfg: EnvConfig) -> tuple[Action, State]:
    """Strict parse of the model reply; raises ValueError/KeyError/TypeError on anything off."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    d = json.loads(m.group())
    a = d["action"]
    kind = a["kind"]
    if kind not in ("move", "pick", "place"):
        raise ValueError(f"bad action kind: {kind!r}")
    target = a.get("target")
    if kind == "move" and target not in cfg.zones:
        raise ValueError(f"bad move target: {target!r}")
    if kind == "pick" and not (
        isinstance(target, int) and not isinstance(target, bool) and 0 <= target < cfg.n_boxes
    ):
        raise ValueError(f"bad pick target: {target!r}")
    if kind == "place":
        target = None
    b = d["believed_next_state"]
    robot_zone = b["robot_zone"]
    box_zones = b["box_zones"]
    if robot_zone not in cfg.zones:
        raise ValueError(f"bad robot_zone: {robot_zone!r}")
    if len(box_zones) != cfg.n_boxes:
        raise ValueError(f"expected {cfg.n_boxes} box_zones, got {len(box_zones)}")
    if any(z != HAND and z not in cfg.zones for z in box_zones):
        raise ValueError(f"bad box_zones: {box_zones!r}")
    return Action(kind, target), State(robot_zone, tuple(box_zones))


class CachedLLM:
    """LiteLLM wrapper with a content-addressed disk cache. Cache hit -> zero network."""

    def __init__(self, model: str, cache_dir: Path):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def complete(self, messages: list[dict]) -> dict:
        """Returns {"text", "tokens_in", "tokens_out", "cache_hit"}."""
        key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "messages": messages, "temperature": 0}, sort_keys=True
            ).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text()) | {"cache_hit": True}
        resp = litellm.completion(model=self.model, messages=messages, temperature=0)
        out = {
            "text": resp.choices[0].message.content,
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens,
        }
        path.write_text(json.dumps(out))
        self.misses += 1
        return out | {"cache_hit": False}


def _blank_meta() -> dict:
    return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cache_hits": 0, "n_parse_retries": 0}


def _accumulate(meta: dict, other: dict) -> None:
    for k in meta:
        meta[k] += other[k]


class Planner:
    """Stepwise JSON-protocol planner with bounded retries on malformed replies."""

    def __init__(self, llm: CachedLLM, cfg: EnvConfig, max_parse_retries: int = 2):
        self.llm = llm
        self.cfg = cfg
        self.max_parse_retries = max_parse_retries

    def draft(
        self, task: Task, history: list[str], note: str | None
    ) -> tuple[Action | None, State | None, dict]:
        """One drafted (action, belief). (None, None, meta) means parse failure."""
        messages = [
            {"role": "system", "content": system_prompt(self.cfg)},
            {"role": "user", "content": user_message(task, history, note)},
        ]
        meta = _blank_meta()
        for attempt in range(1 + self.max_parse_retries):
            r = self.llm.complete(messages)
            meta["calls"] += 1
            meta["tokens_in"] += r["tokens_in"]
            meta["tokens_out"] += r["tokens_out"]
            meta["cache_hits"] += int(r["cache_hit"])
            try:
                action, belief = parse_reply(r["text"], self.cfg)
                return action, belief, meta
            except (ValueError, KeyError, TypeError) as e:
                meta["n_parse_retries"] = min(attempt + 1, self.max_parse_retries)
                messages += [
                    {"role": "assistant", "content": r["text"]},
                    {
                        "role": "user",
                        "content": f"Invalid reply ({e}). Reply with exactly one JSON object "
                        "in the required format.",
                    },
                ]
        return None, None, meta
