# Lucid — W4 Design: Live Grounding Monitor and the `@grounded` Wrapper

**Date:** 2026-07-16
**Status:** Approved
**Scope:** The final milestone. Two deliverables: (1) a live grounding monitor — a FastAPI + websockets app that streams belief-vs-predicted-vs-true state per step from the W2/W3 episode logs (the README demo GIF); (2) `@grounded`, a universal wrapper that adds the doubt gate + grounding logs to any tool-using agent via a small state adapter. No new eval API spend — the monitor replays existing parquet logs.

## Context

W1 built the world model, W2 the hallucinating planner + metric, W3 the adaptive gate + frontier. Everything downstream already consumes the per-step log rows written to `data/agent/<arm>.parquet` (columns: episode_id, step, true_state, believed_state, wm_predicted_state, action, valid, n_revisions, gate_decision, doubt, …). W4 makes that data *visible* and makes the grounding mechanism *reusable*.

Decisions from brainstorming:
- Monitor **replays committed parquet logs** over a websocket (no live agent run, no API spend).
- Frontend is one dependency-free HTML file, dark theme, full-redraw JS.
- `@grounded` is genuinely domain-general via a `state_adapter` seam (the first `Protocol` in the codebase — the W1 plan deferred protocols "until a second implementation exists"; this is that moment).
- Wrapper writes the same parquet schema the monitor reads; no live-streaming of running agents (YAGNI).

## Architecture and data flow

Two new modules plus one static file and one CLI:

- **`src/lucid/monitor.py`** — FastAPI app. `GET /` serves `static/monitor.html`; `GET /api/episodes` lists (arm, episode_id) pairs found in `data/agent/*.parquet`; `GET /api/config` returns the ordered variable names for rendering; `WS /ws/{arm}/{episode_id}` replays that episode's rows as a stream of `StepEvent` JSON messages at a client-controlled frame rate, then sends a `{"done": true}` sentinel and closes.
- **`src/lucid/grounded.py`** — the `@grounded` decorator, `StateAdapter` protocol, `GroundingSession`, and the default warehouse adapter.
- **`static/monitor.html`** — the belief-vs-reality UI (inline CSS/JS, no build).
- **`lucid-monitor` CLI** — a uvicorn launcher.

Data flow (monitor): browser → `GET /api/episodes` → pick one → open `WS` → server reads the parquet rows for that episode, emits one `StepEvent` per row → page redraws each step → done sentinel.

Data flow (`@grounded`): user's `step(state, action) -> belief` → wrapper runs it → `transition_model.predict(state, action)` → `DoubtGate` triage (W3 logic, unchanged) → `StepEvent` appended to `GroundingSession` → wrapper returns original or WM-corrected belief per decision. `session.to_parquet(path)` writes the monitor-compatible schema.

## StepEvent schema

`StepEvent` (frozen dataclass in `monitor.py`) is the wire format the server sends and the page renders. Built from a parquet row:

```
StepEvent(
    step: int,
    action: str,                 # action JSON as logged
    valid: bool,
    variables: list[VarView],    # one per state variable
    grounding: float,            # fraction of variables where belief == truth (the meter)
    gate_decision: str,          # "revise" | "adopt" | "ignore" | ""
    doubt: float,                # -1.0 when no disagreement
)
VarView(name: str, believed: str, predicted: str, truth: str, agree: bool)
```

`agree` is `believed == truth` (per-variable honesty). `grounding` = mean(agree) over variables — the headline "is the agent lying right now" gauge. `predicted` comes from `wm_predicted_state` when present (a disagreement occurred), else equals the truth slot (the WM implicitly agreed). States are decoded from their JSON via the existing `State.from_json`; variable names/order come from `State` (`robot_zone`, then `box_0…`). A helper `step_events(table)` yields `StepEvent`s for an episode's rows; the monitor and any test share it.

## Frontend (`static/monitor.html`)

One file, inline CSS + vanilla JS, no dependencies. On load: `GET /api/episodes` fills a dropdown; **Play** opens the websocket and renders each `StepEvent`.

Layout: a header strip (episode id, step, action, valid/invalid); a prominent **grounding meter** (0–100%, green→red); a **gate badge** (decision + doubt); then the core view — one row per variable with three cells, **Agent believes / World model predicts / Ground truth** — so divergence reads left-to-right. Matching rows render calm (green); a belief≠truth row flashes red and drops the meter. Controls: Play/Pause and a speed slider (sets `fps`, sent as a query param on the ws URL). Dark theme, high-contrast red/green chosen to survive GIF quantization. `ponytail:` note: full-redraw per step (cheap here; no virtual DOM).

## `@grounded` wrapper (`src/lucid/grounded.py`)

```
class StateAdapter(Protocol):
    def variables(self, state) -> dict[str, str]: ...   # flatten any state to named vars
    def goal_vars(self, state, goals) -> set[str]: ...  # which vars are goal-relevant

def grounded(transition_model, adapter, gate=None, session=None):
    """Decorator factory. Wraps step(state, action) -> belief."""
```

The wrapped call: run the user step → `pred = transition_model.predict(state, action)` → compute disputed vars via the adapter (`variables(pred.point_next_state)` vs `variables(belief)`) → if they differ and a gate is present, `gate.decide(pred, disputed, adapter.goal_vars(state, goals))` → on `adopt` return `pred.point_next_state`, on `ignore` or `revise` return `belief` (the wrapper does not re-invoke the user step; revision is the host agent's concern — the wrapper records the decision and corrects tracking only on `adopt`), always append a monitor-schema row (plain dict — `grounded.py` never imports the FastAPI module) to `session`. Returns the (possibly corrected) belief.

**Required refactor for genuine generality:** `DoubtGate.decide` currently derives goal variables from a warehouse-shaped `goals: dict[int, str]` (`{"robot_zone"} | {f"box_{i}" for i in goals}`). W4 changes its third parameter to `goal_vars: set[str]`, with callers computing the set: `run_episode` builds the warehouse set exactly as `decide` does today (behavior unchanged, W3 tests updated mechanically), and the wrapper gets it from the adapter. This is the one W3 touch-point, and it is what makes "universal" true rather than aspirational.

**Generality seam:** with that refactor, `DoubtGate` and `disputed_vars` operate purely on `Prediction` + variable-name sets; only the flattening of a concrete state to named variables is domain-specific, and that lives entirely in the adapter. Warehouse ships `WarehouseAdapter` built from `State`. `GroundingSession` accumulates rows and `.to_parquet(path)` writes the monitor schema.

## Testing and CI

Add `fastapi` + `uvicorn[standard]` (brings `websockets`). CI stays zero-network via FastAPI's in-process `TestClient` (websocket-capable).

- `StepEvent`/`VarView` construction from a crafted row; `grounding` aggregation math; `step_events(table)` sequence.
- `GET /api/episodes` and `/api/config` against a tiny parquet fixture; `GET /` serves the HTML (smoke).
- `WS /ws/{arm}/{episode}` via `TestClient`: assert the `StepEvent` sequence matches the fixture rows and the `{"done": true}` sentinel closes it.
- `@grounded` on the warehouse agent: injected lie → gate fires → correct `StepEvent` logged → corrected belief on adopt.
- **Generality test:** a toy non-warehouse agent (small counter/gridworld with its own state type + a 5-line adapter) → assert a `StepEvent` is logged and the gate triages. This test backs the "universal" claim.
- No browser automation (YAGNI).

## Deliverables

- `lucid-monitor` CLI (uvicorn launcher on `data/agent`).
- README "Live monitor + `@grounded`" section: how to run the monitor, the demo **GIF** (recorded manually from a replayed hallucinating episode — the one artifact not generable headless; the plan documents where it goes and how to record it), and a code snippet applying `@grounded` to a non-warehouse agent.

## Success criteria

1. `lucid-monitor` serves the page and streams a real logged episode end-to-end (manual check; the websocket replay is covered by an automated `TestClient` test).
2. The `@grounded` generality test passes: the wrapper triages a non-warehouse agent's state through the gate and logs a `StepEvent`, using only a small adapter.
3. All tests zero-network; CI green.
4. README leads with the monitor GIF and the frontier chart — the two artifacts that tell the whole project story in images.

## Ponytail deviations

- Parquet replay only; no live-streaming of a running agent.
- No timeline scrubbing; Play/Pause + speed only.
- No JS framework; full-redraw rendering.
- `state_adapter` as the single generality seam (no per-domain gate subclasses).
- The wrapper records decisions and corrects tracking but does not itself re-invoke the user step on `revise` — revision remains the host agent's responsibility (as in W2/W3, where the planner owns the redraft).

## Out of scope

- Live-streaming a running agent to the monitor (future work).
- Auth, multi-user sessions, persistence beyond the parquet logs.
