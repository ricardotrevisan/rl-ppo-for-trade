#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "data" / "mcp_state"
TRIALS_DIR = STATE_DIR / "trials"
LEADERBOARD_CSV = STATE_DIR / "leaderboard.csv"
STATE_JSON = STATE_DIR / "state.json"


METRIC_PATTERNS = {
    "ret_total_pct": re.compile(r"Retorno total:\s+([-+]?\d+(?:\.\d+)?)%"),
    "ret_mean_daily_pct": re.compile(r"Retorno médio diário:\s+([-+]?\d+(?:\.\d+)?)%"),
    "vol_annual_pct": re.compile(r"Volatilidade anual:\s+([-+]?\d+(?:\.\d+)?)%"),
    "sharpe": re.compile(r"Sharpe Ratio:\s+([-+]?\d+(?:\.\d+)?|nan|NaN|inf|-inf)"),
    "max_drawdown_pct": re.compile(r"Máximo Drawdown:\s+([-+]?\d+(?:\.\d+)?)%"),
    "bh_ret_pct": re.compile(r"Retorno buy-and-hold \(slice\):\s+([-+]?\d+(?:\.\d+)?)%"),
    "actions": re.compile(r"Ações escolhidas:\s+hold=(\d+),\s*buy=(\d+),\s*sell=(\d+)"),
    "trades": re.compile(r"Trades efetivados \(durante avaliação\):\s+(\d+)"),
}


def _ensure_dirs() -> None:
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "trades").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "graphs").mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    _ensure_dirs()
    if STATE_JSON.exists():
        try:
            return json.loads(STATE_JSON.read_text())
        except Exception:
            pass
    state = {
        "created_at": dt.datetime.now().isoformat(),
        "pinned_trial_id": None,
        "budgets": {
            "max_trials": 100,
            "per_trial_timeout_minutes": 180,
        },
        "objective": {
            "primary": "sharpe",
            "secondary": "ret_total_pct",
            "max_drawdown_target_pct": 25.0,
        },
    }
    _save_state(state)
    return state


def _save_state(state: Dict[str, Any]) -> None:
    STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _now_id(prefix: str = "trial") -> str:
    return f"{prefix}_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _parse_metrics(text: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for key, pat in METRIC_PATTERNS.items():
        if key == "actions":
            mo = pat.search(text)
            if mo:
                m["actions_hold"] = int(mo.group(1))
                m["actions_buy"] = int(mo.group(2))
                m["actions_sell"] = int(mo.group(3))
            continue
        mo = pat.search(text)
        if mo:
            val = mo.group(1)
            if key == "sharpe":
                try:
                    m[key] = float(val)
                except Exception:
                    m[key] = float("nan")
            else:
                m[key] = float(val)
    mo = METRIC_PATTERNS["trades"].search(text)
    if mo:
        m["trades_executed"] = int(mo.group(1))
    return m


def _rank_key(row: Dict[str, Any]) -> Tuple:
    # Higher sharpe, then total return; lower drawdown
    return (
        float(row.get("sharpe", float("nan"))),
        float(row.get("ret_total_pct", float("nan"))),
        -float(row.get("max_drawdown_pct", float("nan"))),
    )


def _snapshot_outputs() -> set[Path]:
    base = [ROOT / "data" / "metrics", ROOT / "data" / "trades", ROOT / "data" / "graphs"]
    files: set[Path] = set()
    for b in base:
        for p in b.glob("**/*"):
            if p.is_file():
                files.add(p)
    return files


def validate_params(params: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errs: List[str] = []
    p = dict(params)
    # Bounds and types
    def _in(key, lo, hi):
        v = p.get(key)
        if v is None:
            return
        try:
            fv = float(v)
        except Exception:
            errs.append(f"{key} must be numeric")
            return
        if fv < lo or fv > hi:
            errs.append(f"{key} out of range [{lo},{hi}]: {fv}")

    def _posint(key):
        v = p.get(key)
        if v is None:
            return
        try:
            iv = int(v)
            if iv <= 0:
                errs.append(f"{key} must be > 0")
        except Exception:
            errs.append(f"{key} must be int")

    _posint("window_size")
    _posint("risk_window")
    _posint("lot_size")
    _in("max_trade_fraction", 0.01, 0.5)
    _in("dd_penalty", 0.0, 1.0)
    _in("turnover_penalty", 0.0, 0.1)
    _in("loss_penalty", 0.0, 1.0)
    _in("inv_mom_penalty", 0.0, 1.0)
    _in("sell_turnover_factor", 0.0, 2.0)
    _in("learning_rate", 1e-6, 1e-2)
    _posint("n_steps")
    _posint("batch_size")
    _in("ent_coef", 0.0, 0.2)
    _in("clip_range", 0.05, 0.5)
    # Reward mode
    if "reward_mode" in p and p["reward_mode"] not in ("log", "risk_adj"):
        errs.append("reward_mode must be 'log' or 'risk_adj'")
    return (len(errs) == 0, errs, p)


def build_adaptation_cmd(params: Dict[str, Any], fast_mode: bool = True) -> List[str]:
    uv = shutil.which("uv")
    base = [uv, "run", "adaptation.py"] if uv else [sys.executable, "adaptation.py"]
    # Map params to CLI
    def arg(name: str, val: Any) -> List[str]:
        return [f"--{name}", str(val)]

    cmd = list(base)
    # Required windows
    for k in ("window_size", "risk_window", "train_start", "train_end", "eval_start", "eval_end"):
        if k in params and params[k] is not None:
            cmd += arg(k.replace("_", "-"), params[k])
    # Reward/behavior
    for k in (
        "reward_mode",
        "dd_penalty",
        "turnover_penalty",
        "loss_penalty",
        "inv_mom_penalty",
        "sell_turnover_factor",
        "max_trade_fraction",
        "lot_size",
    ):
        if k in params and params[k] is not None:
            cmd += arg(k.replace("_", "-"), params[k])
    # PPO knobs
    for k in ("learning_rate", "batch_size", "n_steps", "n_epochs", "ent_coef", "clip_range"):
        if k in params and params[k] is not None:
            cmd += arg(k.replace("_", "-"), params[k])
    # Mode and misc
    cmd += ["--mode", "train"]
    if params.get("downside_only"):
        cmd.append("--downside-only")
    # Seed and envs
    if params.get("seed") is not None:
        cmd += ["--seed", str(int(params["seed"]))]

    if fast_mode:
        cmd += ["--total-timesteps", str(int(params.get("total_timesteps_fast", 40_000)))]
        cmd += ["--num-envs", "1", "--deterministic"]
    else:
        if params.get("total_timesteps"):
            cmd += ["--total-timesteps", str(int(params["total_timesteps"]))]
        if params.get("num_envs"):
            cmd += ["--num-envs", str(int(params["num_envs"]))]

    cmd.append("--print-params")
    return cmd


def _read_text_limited(path: Path, limit: int = 2_000_000) -> str:
    try:
        data = path.read_bytes()
        return data[:limit].decode(errors="replace")
    except Exception:
        return ""


def run_trial(params: Dict[str, Any], tag: Optional[str] = None, fast_mode: bool = True, timeout_minutes: int = 180) -> Dict[str, Any]:
    _ensure_dirs()
    ok, errs, validated = validate_params(params)
    if not ok:
        return {"status": "invalid", "errors": errs}
    trial_id = _now_id()
    tdir = TRIALS_DIR / trial_id
    tdir.mkdir(parents=True, exist_ok=True)

    cmd = build_adaptation_cmd(validated, fast_mode=fast_mode)
    config = {
        "params": validated,
        "fast_mode": bool(fast_mode),
        "tag": tag,
        "cmd": cmd,
        "created_at": dt.datetime.now().isoformat(),
    }
    (tdir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    before = _snapshot_outputs()
    try:
        run = subprocess.run(
            cmd,
            input="1\n",
            text=True,
            capture_output=True,
            timeout=timeout_minutes * 60,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "trial_id": trial_id}
    after = _snapshot_outputs()
    new_files = sorted([str(p.relative_to(ROOT)) for p in after - before])

    (tdir / "stdout.txt").write_text(run.stdout or "")
    (tdir / "stderr.txt").write_text(run.stderr or "")
    metrics = _parse_metrics((run.stdout or "") + "\n" + (run.stderr or ""))
    (tdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Update leaderboard
    row = {
        "trial_id": trial_id,
        "timestamp": dt.datetime.now().isoformat(),
        "tag": tag or "",
        "fast_mode": int(bool(fast_mode)),
        **metrics,
    }
    new_file = not LEADERBOARD_CSV.exists()
    with LEADERBOARD_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "timestamp",
                "tag",
                "fast_mode",
                "ret_total_pct",
                "ret_mean_daily_pct",
                "vol_annual_pct",
                "sharpe",
                "max_drawdown_pct",
                "bh_ret_pct",
                "actions_hold",
                "actions_buy",
                "actions_sell",
                "trades_executed",
            ],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    return {
        "status": "ok" if run.returncode == 0 else f"exit_{run.returncode}",
        "trial_id": trial_id,
        "metrics": metrics,
        "artifacts": new_files,
        "stdout_path": str((tdir / "stdout.txt").relative_to(ROOT)),
        "stderr_path": str((tdir / "stderr.txt").relative_to(ROOT)),
        "config_path": str((tdir / "config.json").relative_to(ROOT)),
    }


def list_trials(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not LEADERBOARD_CSV.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with LEADERBOARD_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast numbers
            for k in (
                "ret_total_pct",
                "ret_mean_daily_pct",
                "vol_annual_pct",
                "sharpe",
                "max_drawdown_pct",
                "bh_ret_pct",
                "actions_hold",
                "actions_buy",
                "actions_sell",
                "trades_executed",
            ):
                if r.get(k) not in (None, ""):
                    try:
                        r[k] = float(r[k]) if k not in ("actions_hold","actions_buy","actions_sell","trades_executed") else int(float(r[k]))
                    except Exception:
                        pass
            rows.append(r)
    rows.sort(key=_rank_key, reverse=True)
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def get_best() -> Optional[Dict[str, Any]]:
    rows = list_trials(limit=1)
    return rows[0] if rows else None


def pin_best(trial_id: str) -> Dict[str, Any]:
    state = _load_state()
    state["pinned_trial_id"] = trial_id
    _save_state(state)
    return {"status": "ok", "pinned_trial_id": trial_id}


def get_state() -> Dict[str, Any]:
    return _load_state()


def read_trial_config(trial_id: str) -> Optional[Dict[str, Any]]:
    p = TRIALS_DIR / trial_id / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def read_trial_metrics(trial_id: str) -> Optional[Dict[str, Any]]:
    p = TRIALS_DIR / trial_id / "metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def main_cli():
    parser = argparse.ArgumentParser(description="Optimizer core utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate")
    v.add_argument("--params", required=True, help="Path to params JSON")

    r = sub.add_parser("run")
    r.add_argument("--params", required=True)
    r.add_argument("--tag", default=None)
    r.add_argument("--full", action="store_true", help="Run full budget (disable fast mode)")
    r.add_argument("--timeout-minutes", type=int, default=120)

    l = sub.add_parser("list")
    l.add_argument("--top", type=int, default=10)

    b = sub.add_parser("best")

    p = sub.add_parser("pin")
    p.add_argument("trial_id")

    args = parser.parse_args()
    if args.cmd == "validate":
        params = json.loads(Path(args.params).read_text())
        ok, errs, out = validate_params(params)
        print(json.dumps({"ok": ok, "errors": errs, "params": out}, indent=2, ensure_ascii=False))
    elif args.cmd == "run":
        params = json.loads(Path(args.params).read_text())
        res = run_trial(params, tag=getattr(args, "tag", None), fast_mode=not args.full, timeout_minutes=args.timeout_minutes)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "list":
        rows = list_trials(limit=args.top)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.cmd == "best":
        print(json.dumps(get_best(), indent=2, ensure_ascii=False))
    elif args.cmd == "pin":
        print(json.dumps(pin_best(args.trial_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main_cli()

