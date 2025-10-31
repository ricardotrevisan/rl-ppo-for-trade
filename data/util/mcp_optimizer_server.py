#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Optional
from pathlib import Path

from fastmcp import FastMCP
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import math

# Reuse optimizer core helpers
HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import optimizer_core as core  # type: ignore


mcp = FastMCP("optimizer")
_executor = ThreadPoolExecutor(max_workers=2)
_futures: Dict[str, object] = {}


def _sanitize(obj):
    """Recursively replace NaN/inf with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# Resources
@mcp.resource("mcp://optimizer/state")
def resource_state() -> str:
    return json.dumps(core.get_state(), indent=2, ensure_ascii=False)


@mcp.resource("mcp://optimizer/leaderboard")
def resource_leaderboard() -> str:
    p = HERE.parents[2] / "data" / "mcp_state" / "leaderboard.csv"
    return p.read_text() if p.exists() else ""


@mcp.resource("mcp://optimizer/trial/{trial_id}/config")
def resource_trial_config(trial_id: str) -> str:
    data = core.read_trial_config(trial_id) or {}
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.resource("mcp://optimizer/trial/{trial_id}/metrics")
def resource_trial_metrics(trial_id: str) -> str:
    data = core.read_trial_metrics(trial_id) or {}
    return json.dumps(data, indent=2, ensure_ascii=False)


# Tools
@mcp.tool()
def optimizer_run_trial(
    params: dict,
    tag: Optional[str] = None,
    fast_mode: bool = True,
    timeout_minutes: int = 180,
) -> dict:
    """Run adaptation.py with params and return metrics/artifacts."""
    res = core.run_trial(params, tag=tag, fast_mode=fast_mode, timeout_minutes=timeout_minutes)
    return _sanitize(res)


@mcp.tool()
def optimizer_validate_params(params: dict) -> dict:
    """Validate and normalize params without running."""
    ok, errs, out = core.validate_params(params)
    if ok:
        return {"status": "ok", "params": out}
    else:
        return {"status": "invalid", "reasons": errs, "params": out}


@mcp.tool()
def optimizer_list_trials(top: Optional[int] = None) -> dict:
    """List trials sorted by objective."""
    return _sanitize({"trials": core.list_trials(limit=top)})


@mcp.tool()
def optimizer_get_best() -> dict:
    """Get current best trial summary."""
    return _sanitize({"best": core.get_best()})


@mcp.tool()
def optimizer_pin_best(trial_id: str) -> dict:
    """Pin a specific trial as best."""
    return core.pin_best(trial_id)


@mcp.tool()
def optimizer_start_trial(
    params: dict,
    tag: Optional[str] = None,
    fast_mode: bool = True,
    timeout_minutes: int = 180,
) -> dict:
    """Start a trial asynchronously and return the trial_id immediately."""
    trial_id = core._now_id()  # type: ignore[attr-defined]
    # Submit background task
    fut = _executor.submit(core.run_trial, params, tag, fast_mode, timeout_minutes, trial_id)
    _futures[trial_id] = fut
    return {"status": "started", "trial_id": trial_id}


@mcp.tool()
def optimizer_poll_trial(trial_id: str) -> dict:
    """Poll the status of a previously started trial."""
    # If we have a future, check its status
    fut = _futures.get(trial_id)
    if fut is not None:
        if getattr(fut, "done", lambda: False)():
            try:
                res = fut.result()
                return {"status": "done", "result": _sanitize(res)}
            finally:
                # Allow GC
                _futures.pop(trial_id, None)
        else:
            return {"status": "running"}
    # Fallback to filesystem status for processes started in a prior session
    status = core.read_trial_status(trial_id)
    if status and status.get("status") and status.get("status") != "running":
        res = core.read_trial_result(trial_id) or {}
        return {"status": "done", "result": _sanitize(res)}
    return {"status": "running"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8008)
