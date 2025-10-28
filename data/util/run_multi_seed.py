#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
import shutil

try:
    import numpy as np
except Exception:  # fallback if numpy not available at import time
    np = None  # type: ignore


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


def parse_metrics(text: str) -> dict:
    m = {}
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
    # trades last
    mo = METRIC_PATTERNS["trades"].search(text)
    if mo:
        m["trades_executed"] = int(mo.group(1))
    return m


def np_safe_stats(values):
    # returns dict of mean, std, median, p5, p95 (nan-safe for sharpe)
    arr = [v for v in values if v is not None]
    if not arr:
        return {"mean": None, "std": None, "median": None, "p5": None, "p95": None}
    if np is None:
        # fall back without percentiles
        try:
            return {
                "mean": mean(arr),
                "std": None,
                "median": median(arr),
                "p5": None,
                "p95": None,
            }
        except Exception:
            return {"mean": None, "std": None, "median": None, "p5": None, "p95": None}
    a = np.array(arr, dtype=float)
    return {
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.nanmedian(a)),
        "p5": float(np.nanpercentile(a, 5)),
        "p95": float(np.nanpercentile(a, 95)),
    }


def build_adaptation_cmd(args, seed: int) -> list[str]:
    uv_path = shutil.which("uv")
    if uv_path:
        base = [uv_path, "run", "adaptation.py"]
    else:
        # Fallback: run directly with current interpreter
        base = [sys.executable, "adaptation.py"]
    cmd = base + [
        "--window-size",
        str(args.window_size),
        "--risk-window",
        str(args.risk_window),
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--eval-start",
        args.eval_start,
        "--eval-end",
        args.eval_end,
        "--dd-penalty",
        str(args.dd_penalty),
        "--turnover-penalty",
        str(args.turnover_penalty),
        "--loss-penalty",
        str(args.loss_penalty),
        "--inv-mom-penalty",
        str(args.inv_mom_penalty),
        "--sell-turnover-factor",
        str(args.sell_turnover_factor),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(seed),
    ]
    if args.downside_only:
        cmd.append("--downside-only")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple seeds for adaptation.py and summarize results")
    # seed control
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to run (seeds = start-seed..start-seed+num-seeds-1)")
    g.add_argument("--seeds", type=int, nargs="*", help="Explicit list of seeds to run")
    parser.add_argument("--start-seed", type=int, default=0, help="Start seed when using --num-seeds")

    # pass-through to adaptation
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--risk-window", type=int, default=20)
    parser.add_argument("--train-start", type=str, default="2022-01-01")
    parser.add_argument("--train-end", type=str, default="2023-01-01")
    parser.add_argument("--eval-start", type=str, default="2023-01-02")
    parser.add_argument("--eval-end", type=str, default="2025-12-31")
    parser.add_argument("--downside-only", action="store_true")
    parser.add_argument("--dd-penalty", type=float, default=0.05)
    parser.add_argument("--turnover-penalty", type=float, default=0.002)
    parser.add_argument("--loss-penalty", type=float, default=0.10)
    parser.add_argument("--inv-mom-penalty", type=float, default=0.02)
    parser.add_argument("--sell-turnover-factor", type=float, default=0.5)
    parser.add_argument("--num-envs", type=int, default=8)

    # output control
    parser.add_argument("--out-dir", type=str, default="data/metrics", help="Where to write summary CSVs")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag to include in output filenames")

    args = parser.parse_args()

    if args.seeds is None:
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    else:
        seeds = list(dict.fromkeys(args.seeds))  # de-dup preserve order

    print(f"Running seeds: {seeds}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"multi_seed_{stamp}" + (f"_{args.tag}" if args.tag else "")

    per_seed_rows = []
    for s in seeds:
        cmd = build_adaptation_cmd(args, s)
        print("\n=== Running seed:", s)
        print(f"cmd> {cmd}")
        try:
            # Feed '1' to choose training path
            run = subprocess.run(
                cmd,
                input="1\n",
                text=True,
                capture_output=True,
                check=False,
            )
        except KeyboardInterrupt:
            print("Interrupted by user.")
            return 1
        except Exception as e:
            print(f"Failed to run seed {s}: {e}")
            continue

        stdout = run.stdout or ""
        stderr = run.stderr or ""
        if run.returncode != 0:
            print(f"Seed {s} exited with code {run.returncode}")
            if stderr:
                print(stderr[-2000:])
        metrics = parse_metrics(stdout + "\n" + stderr)
        metrics["seed"] = s
        per_seed_rows.append(metrics)

    # Write per-seed CSV
    per_seed_path = out_dir / f"{base}.csv"
    fieldnames = [
        "seed",
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
    ]
    with per_seed_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"Per-seed metrics: {per_seed_path}")

    # Summary CSV
    def col(vals, key):
        xs = [v.get(key) for v in vals]
        return [x for x in xs if x is not None]

    summary = {
        "seeds": len(per_seed_rows),
    }
    for key in [
        "ret_total_pct",
        "ret_mean_daily_pct",
        "vol_annual_pct",
        "sharpe",
        "max_drawdown_pct",
        "bh_ret_pct",
        "trades_executed",
    ]:
        vals = col(per_seed_rows, key)
        try:
            stats = np_safe_stats(vals)
        except Exception:
            stats = {"mean": None, "std": None, "median": None, "p5": None, "p95": None}
        for sfx, v in stats.items():
            summary[f"{key}_{sfx}"] = v

    summary_path = out_dir / f"{base}_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])
    print(f"Summary: {summary_path}")

    # Also print brief summary
    for k in ["ret_total_pct", "sharpe", "max_drawdown_pct"]:
        print(
            f"{k}: mean={summary.get(k + '_mean')} std={summary.get(k + '_std')} "
            f"median={summary.get(k + '_median')} p5={summary.get(k + '_p5')} p95={summary.get(k + '_p95')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
