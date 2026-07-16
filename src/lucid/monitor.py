"""Live grounding monitor: replay episode logs as belief-vs-reality streams."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse


@dataclass(frozen=True)
class VarView:
    name: str
    believed: str
    predicted: str
    truth: str
    agree: bool


@dataclass(frozen=True)
class StepEvent:
    """Wire format the monitor streams; one per executed step."""

    step: int
    action: str
    valid: bool
    variables: list[VarView]
    grounding: float  # fraction of variables where belief == truth
    gate_decision: str
    doubt: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _to_vars(state_json: str) -> dict[str, str]:
    """Flatten a logged state JSON to named variables (warehouse State or flat dict)."""
    d = json.loads(state_json)
    if "box_zones" in d:
        return {"robot_zone": d["robot_zone"]} | {
            f"box_{i}": z for i, z in enumerate(d["box_zones"])
        }
    return {str(k): str(v) for k, v in d.items()}


def step_event(row: dict) -> StepEvent:
    truth = _to_vars(row["true_state"])
    believed = _to_vars(row["believed_state"]) if row["believed_state"] else truth
    predicted = _to_vars(row["wm_predicted_state"]) if row["wm_predicted_state"] else truth
    variables = [
        VarView(n, believed.get(n, "?"), predicted.get(n, "?"), t, believed.get(n) == t)
        for n, t in truth.items()
    ]
    return StepEvent(
        step=row["step"],
        action=row["action"] or "",
        valid=row["valid"],
        variables=variables,
        grounding=sum(v.agree for v in variables) / len(variables),
        gate_decision=row["gate_decision"],
        doubt=row["doubt"],
    )


def step_events(rows: list[dict]) -> list[StepEvent]:
    """Ordered events for one episode; parse-failure rows carry no belief and are skipped."""
    live = [r for r in rows if not r["parse_failure"]]
    return [step_event(r) for r in sorted(live, key=lambda r: r["step"])]


def create_app(data_dir: Path | str = "data/agent") -> FastAPI:
    data_dir = Path(data_dir)
    app = FastAPI(title="Lucid grounding monitor")

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "monitor.html")

    @app.get("/api/episodes")
    def episodes():
        out = []
        for f in sorted(data_dir.glob("*.parquet")):
            ids = pq.read_table(f, columns=["episode_id"]).column("episode_id").to_pylist()
            out.append({"arm": f.stem, "episodes": sorted(set(ids))})
        return out

    @app.get("/api/config")
    def config():
        files = sorted(data_dir.glob("*.parquet"))
        if not files:
            return {"variables": []}
        row = pq.read_table(files[0]).to_pylist()[0]
        return {"variables": list(_to_vars(row["true_state"]))}

    @app.websocket("/ws/{arm}/{episode_id}")
    async def replay(ws: WebSocket, arm: str, episode_id: int):
        await ws.accept()
        fps = float(ws.query_params.get("fps", 3))
        rows = [
            r
            for r in pq.read_table(data_dir / f"{arm}.parquet").to_pylist()
            if r["episode_id"] == episode_id
        ]
        for ev in step_events(rows):
            await ws.send_text(ev.to_json())
            await asyncio.sleep(1 / fps)
        await ws.send_text(json.dumps({"done": True}))
        await ws.close()

    return app


app = create_app()
