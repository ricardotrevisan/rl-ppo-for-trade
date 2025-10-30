#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Optional
from pathlib import Path

from fastmcp import FastMCP

# Reuse optimizer core helpers
HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import optimizer_core as core  # type: ignore


mcp = FastMCP("optimizer")


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
    return core.run_trial(params, tag=tag, fast_mode=fast_mode, timeout_minutes=timeout_minutes)


@mcp.tool()
def optimizer_validate_params(params: dict) -> dict:
    """Validate and normalize params without running."""
    ok, errs, out = core.validate_params(params)
    return {"ok": ok, "errors": errs, "params": out}


@mcp.tool()
def optimizer_list_trials(top: Optional[int] = None) -> dict:
    """List trials sorted by objective."""
    return {"trials": core.list_trials(limit=top)}


@mcp.tool()
def optimizer_get_best() -> dict:
    """Get current best trial summary."""
    return {"best": core.get_best()}


@mcp.tool()
def optimizer_pin_best(trial_id: str) -> dict:
    """Pin a specific trial as best."""
    return core.pin_best(trial_id)


if __name__ == "__main__":
    mcp.run()

