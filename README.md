**Project Overview**
- Purpose: Experimental framework for algorithmic trading research, focused on single-asset equity trading using reinforcement learning (PPO/A2C, Stable-Baselines3).
- Objective: Evaluate reward-function design and entropy-driven exploration under stochastic market dynamics, quantifying how agent entropy evolves given different algorithmic parameters and feature sets.
- Environment: Custom Gymnasium environment trading PETR4.SA (B3) using real Yahoo Finance data with offline caching to prevent redundant downloads across workers.
- Market realism:
  - Real timestamps and execution delays
  - Lot-based partial position sizing, capped at 5% of capital/position
  - Per-trade transaction fees
  - No short selling (unecessary complexity at the moment)
  - On-policy normalization of observations
  - Designed for reproducible experimentation on policy stability, entropy dynamics, and reward shaping in realistic equity-trading contexts.

**Key Features**
- Custom `TradingEnv` with realistic constraints (fees, lots, trade caps), optional risk‑adjusted rewards, and observation normalization.
- PPO training (Stable‑Baselines3), with GPU support and parallel envs for speed.
- Simple, scriptable CLI: train/eval modes, core knobs exposed, and utilities for multi‑seed runs and data cleanup.

**Centralized Indicators**
- Single source for indicators in `featureset.py` with on‑disk cache under `data/indicators/`.
- Precomputed per date range to avoid recomputation across workers/runs.
- Implemented indicators:
  - Trend/momentum: `SMA5/20`, `EMA12/26`, `MACD_hist`, `ROC5`
  - Oscillators/regime: `RSI14`, `StochK−D`, `ADX14`
  - Volatility: rolling std of returns over window, `ATR14/price`
  - Derived: return z‑score (`ret_z`), interaction `stoch_adx = stoch_diff * (adx14/100)`
- Active observation features (9), in order:
  - `stoch_diff`, `roll_std`, `atr14_norm`, `adx14`, `macd_hist`, `roc5`, `ret_z`, `rsi14`, `ema_ratio`
- Warm‑up: episodes start at `max(window_size, 35)` so all lookbacks are valid.

**Data Handling**
- On first run, the script caches Yahoo Finance OHLCV to CSV under `data/` so multiple workers don’t re‑download.
- Cache files:
  - `data/PETR4.SA_2022-01-01_2023-01-01.csv` (training range)
  - `data/PETR4.SA_2023-01-02_2025-12-31.csv` (evaluation range)
- To refresh data, delete the cached CSVs; they will be downloaded again automatically.

**Environment & Trading Rules**
- Actions: `0=Hold`, `1=Buy`, `2=Sell`.
- Sizing: buys/sells 1–N lots, capped by `max_trade_fraction` (5%) of available cash or current position; if cap < 1 lot, the trade is skipped.
- No shorts: sells only reduce an existing long; position never goes below 0.
- Fees: charged only when trading, as `transaction_fee_rate * (qty * price)`.
- Reward (configurable):
  - Base: log portfolio return between steps
  - Optional risk adjustment by rolling volatility (downside or full)
  - Small penalties: drawdown, turnover, and negative returns

**GPU Acceleration**
- Automatically selects device in `main()` (CUDA > MPS > CPU).
- Training (Option 1) uses `SubprocVecEnv(n_envs=8)` and a larger PPO batch/arch to benefit from GPU.
- Simulation (Option 2) runs single‑process evaluation (no workers) for simplicity.

**Prerequisites**
- Python 3.9+
- Recommended: a virtual environment
- Packages:
  - `stable-baselines3`, `gymnasium`, `torch` (CUDA build optional), `yfinance`, `pandas`, `numpy`, `matplotlib`
  - `ta-lib` (TA‑Lib) for technical indicators

Notes on TA‑Lib
- TA‑Lib may require system libraries. Common options:
  - Linux: install system `ta-lib` dev package, then `pip install ta-lib`
  - Or use prebuilt wheels (if available for your platform)

**Setup**
- Create and activate a virtual environment:
  - `python -m venv .venv`
  - `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
- Install dependencies (CPU PyTorch example):
  - `pip install --upgrade pip`
  - `pip install stable-baselines3 gymnasium torch yfinance pandas numpy matplotlib ta-lib`
- For CUDA PyTorch, install the matching wheel for your CUDA version (see PyTorch docs), then install the rest.

**Quick Start**
- Train (non‑interactive):
  - `uv run adaptation.py --mode train --train-start 2022-01-01 --train-end 2023-01-01 --eval-start 2023-01-02 --eval-end 2025-12-31`
- Evaluate saved model:
  - `uv run adaptation.py --mode eval --eval-start 2023-01-02 --eval-end 2025-12-31`
- Minimal knobs you may want to change:
  - Data windows: `--train-start/--train-end/--eval-start/--eval-end`
  - Reward base: `--reward-mode {log,risk_adj}` and `--risk-window` (default 20)
  - Trading behavior: `--max-trade-fraction` (default 0.10), `--lot-size` (default 100)
  - Speed/repro: `--num-envs 8` (faster) or `--num-envs 1 --deterministic` (reproducible)
- Print all used params to terminal:
  - Add `--print-params` to any command to dump the full run configuration (CLI args plus derived settings like device and effective env count).

**Hyperparameter Sweep (Stdout + CSV)**
- Random sweep mode to maximize validation Sharpe with short trainings and early stop.
- Run:
  - `uv run adaptation.py --mode sweep --train-start 2022-01-01 --train-end 2023-01-01 --eval-start 2023-01-02 --eval-end 2023-12-31 --sweep-trials 32 --sweep-seeds 2`
- Persists every trial to CSV: `data/metrics/sweep_results_<timestamp>.csv` with:
  - `run_ts`, `device`, `trial`, `mean_sharpe`, `seeds`, `seed_sharpes`, `trial_wall_time_sec`, `sweep_elapsed_sec`
  - PPO/env params: `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `ent_coef`, `clip_range`, `gamma`, `gae_lambda`, `net_arch`, `risk_window`, `downside_only`, `dd_penalty`
- Saves best config JSON to `data/metrics/sweep_best_<timestamp>.json`.

**Offline Feature Report (Stdout Only)**
- Consistent with training features via `featureset.compute_feature_frame`.
- Run:
  - `uv run features_report.py --ticker PETR4.SA --start 2022-01-01 --end 2023-01-01 --window-size 20`
- Prints:
  - Non‑null coverage per feature
  - High‑correlation pairs (Pearson) for redundancy pruning
  - Information Coefficient (Spearman with next return), overall and by splits
  - PCA summary (explained variance, top loadings) for redundancy diagnostics
  - Suggested de‑correlated subset (cap ~12 features)

**Outputs**
- Metrics: CSVs under `data/metrics/` (portfolio, returns, drawdown by run/slice)
- Trades: CSVs under `data/trades/` with per‑trade records (timestamp, side, qty, price, fees, equity before/after, position)
- Graphs: PNGs under `data/graphs/` (equity curve)
- Console: action counts, trades executed, total return, daily mean, annualized vol, Sharpe, max drawdown

**Common CLI Flags**
- Data: `--train-start/--train-end/--eval-start/--eval-end`, `--window-size`, `--risk-window`, `--downside-only`
- Reward: `--reward-mode {log,risk_adj}`, `--dd-penalty`, `--turnover-penalty`, `--loss-penalty`, `--inv-mom-penalty`, `--sell-turnover-factor`
- Trading: `--max-trade-fraction` (cap per trade), `--lot-size` (shares per lot), `--starting-cash` (via code default 100k)
- PPO: `--total-timesteps`, `--learning-rate`, `--batch-size`, `--n-steps`, `--n-epochs`, `--ent-coef`, `--clip-range`, `--gamma`, `--gae-lambda`, `--num-envs`
- Repro: `--seed`, `--deterministic`
 - Diagnostics: `--print-params` (prints all run parameters before executing)

**Realism Considerations**
- Execution timing: current logic fills at the current bar price and marks to next. For stricter realism, you can shift fills to the next open/bar and add a spread/slippage model.
- Liquidity: the 5% cap constrains turnover; you can add ADV‑based caps if needed.

**Troubleshooting**
- No trades: ensure one lot fits the cap (lot_size × price ≤ max_trade_fraction × cash). Adjust `--lot-size` and/or `--max-trade-fraction`.
- TA‑Lib import: install system TA‑Lib or a compatible wheel.
- GPU not used: verify CUDA is available in PyTorch and `nvidia-smi` shows activity.

**Utilities**
- Multi‑seed runner: see `data/util/README.md` for `run_multi_seed.py` (aggregate metrics across seeds).
- Data cleaner: see `data/util/README.md` for `clear_data.py` (remove `.csv`/`.png` under `data/`).
- Optimizer core + CLI:
  - Core library at `data/util/optimizer_core.py` encapsulates trial running, metrics parsing, persistent state (`data/mcp_state/`).
  - CLI wrapper `data/util/optimizerctl.py` provides:
    - Validate params: `python data/util/optimizerctl.py validate --params params.json`
    - Run trial (fast): `python data/util/optimizerctl.py run --params params.json --tag testA`
    - Run full: `python data/util/optimizerctl.py run --params params.json --full --timeout-minutes 240`
    - List top: `python data/util/optimizerctl.py list --top 10`
    - Best summary: `python data/util/optimizerctl.py best`
    - Pin best: `python data/util/optimizerctl.py pin <trial_id>`
  - Artifacts and state are stored under `data/mcp_state/` and standard `data/metrics|trades|graphs` folders.

**MCP Server (fasmcp/fastmcp)**
- Implemented with FastMCP 2.x (PyPI: `fastmcp`). Entry: `data/util/mcp_optimizer_server.py`.
- Resources:
  - `mcp://optimizer/state`
  - `mcp://optimizer/leaderboard`
  - `mcp://optimizer/trial/{trial_id}/config`
  - `mcp://optimizer/trial/{trial_id}/metrics`
- Tools:
  - `optimizer_run_trial(params, tag?, fast_mode?, timeout_minutes?)`
  - `optimizer_start_trial(params, tag?, fast_mode?, timeout_minutes?)` — starts asynchronously and returns `trial_id`
  - `optimizer_poll_trial(trial_id)` — returns `{status: running}` or `{status: done, result: {...}}`
  - `optimizer_validate_params(params)`
  - `optimizer_list_trials(top?)`
  - `optimizer_get_best()`
  - `optimizer_pin_best(trial_id)`
  - Supported PPO params via `params`: `learning_rate`, `batch_size`, `n_steps`, `n_epochs`, `ent_coef`, `clip_range`, `gamma`, `gae_lambda`, plus `total_timesteps`, `num_envs`, and `seed`.
- Run server:
  - `uv run data/util/mcp_optimizer_server.py`
  - Or `python data/util/mcp_optimizer_server.py`
- Point your LLM agent to the stdio server as an MCP endpoint.

Async usage notes
- Long‑running HTTP calls may time out in connectors. Prefer the async pair:
  - Start: `optimizer_start_trial({...})` → `{status: "started", trial_id}` immediately.
  - Poll: `optimizer_poll_trial({trial_id})` until `{status: "done", result}`.
- Status/result files persist under `data/mcp_state/trials/<trial_id>/`:
  - `status.json` — `running` → final status (`ok`, `exit_<code>`, `timeout`)
  - `result.json` — full result dict (metrics, artifacts, paths)
  - `metrics.json` — parsed metrics
  - `config.json` — command and normalized params
  This allows polling to survive server restarts.

**File Map**
- `adaptation.py` — environment, training/evaluation pipeline, caching
- `data/` — cached market data and outputs (`metrics/`, `trades/`, `graphs/`)
