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
- Custom `TradingEnv` (see `adaptation.py:23`) with:
  - Real timestamps from Yahoo data
  - Configurable `starting_cash`, `lot_size`, `max_trade_fraction` (default 5%)
  - No short positions (long/flat only)
  - Per‑trade fees on traded value (`transaction_fee_rate`)
  - Reward design: log return with optional risk adjustment (rolling volatility), and small penalties for drawdown/turnover/losses
  - Observation normalization via `VecNormalize`
- PPO training with optional GPU acceleration (PyTorch CUDA) and parallel environments (`SubprocVecEnv`).
- Deterministic out‑of‑sample simulation (Option 2) with single‑process evaluation.
- Robust trade logging: writes a detailed CSV of executed trades collected via `info["trade"]`.

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

**How To Run**
- Start the script:
  - `python adaptation.py [--window-size N] [--risk-window M] [--train-start YYYY-MM-DD] [--train-end YYYY-MM-DD] [--eval-start YYYY-MM-DD] [--eval-end YYYY-MM-DD]`
    - `--window-size` (default `20`): observation window size used for features
    - `--risk-window` (default `20`): rolling window for risk adjustment in reward
    - `--train-start` (default `2022-01-01`) / `--train-end` (default `2023-01-01`): training data range
    - `--eval-start` (default `2023-01-02`) / `--eval-end` (default `2025-12-31`): evaluation data range
- You will be prompted:
  - `[1] Treinar novo modelo` — trains PPO with 8 parallel envs (uses GPU if available), saves the best model and VecNormalize stats
  - `[2] Carregar e rodar existente` — loads the saved model and runs a deterministic out‑of‑sample simulation (single‑process)

**Outputs**
- `metrics_best_model.csv` — time series of portfolio, returns, drawdown for the simulation period
- `trade_log_detailed.csv` — per‑trade records collected during evaluation, with:
  - timestamp, symbol, side, qty, price, fees, equity_before, equity_after, position_qty_after
- `run_results.png` — portfolio equity curve plot for the simulation
- Console summary — action counts, trades executed, total return, daily mean return, annualized vol, Sharpe, max drawdown

**Configuration**
- Core parameters (set in `TradingEnv` constructor in `adaptation.py`):
  - `starting_cash` (default `100_000.0`)
  - `lot_size` (default `100`)
  - `max_trade_fraction` (default `0.05` → 5%)
  - `transaction_fee_rate` (default `0.0005` → 5 bps)
  - `window_size` (default `20`)
  - Reward knobs: `reward_mode` ("log" or "risk_adj"), `risk_window`, `downside_only`, `dd_penalty`, `turnover_penalty`, `loss_penalty`

**Realism Considerations**
- Execution timing: current logic fills at the current bar price and marks to next. For stricter realism, you can shift fills to the next open/bar and add a spread/slippage model.
- Liquidity: the 5% cap constrains turnover; you can add ADV‑based caps if needed.

**Troubleshooting**
- TA‑Lib import error: ensure the TA‑Lib system library is installed or use a compatible wheel for your platform.
- Empty trade log: see the action counts in console; if the policy held, there may be no trades. Fees and caps also affect trade frequency.
- GPU not used: verify with `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` and check `nvidia-smi` during training.

**File Map**
- `adaptation.py` — environment, training, evaluation pipeline and caching
- `data/` — CSV cache for Yahoo Finance downloads (auto‑created)
