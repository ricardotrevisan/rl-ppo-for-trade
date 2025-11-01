import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import talib
from featureset import Ohlcv, precompute_with_cache
import torch
import os
import random
import warnings
from datetime import datetime, timedelta
import argparse
import json
import csv
import time
import csv
warnings.filterwarnings("ignore")
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass


def get_cached_data(ticker: str, start: str, end: str, cache_dir: str = "data") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace(":", "_").replace("/", "_")
    fname = f"{safe_ticker}_{start}_{end}.csv"
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        print(f"📥 Cache não encontrado. Baixando {ticker} ({start}→{end}) uma vez...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("Sem dados retornados do yfinance para cache.")
        df.to_csv(path)
        print(f"💾 Dados salvos em cache: {path}")
    else:
        print(f"📁 Usando cache existente: {path}")
    return path

# ============================================================
#  Custom Trading Environment using real Yahoo Finance data
# ============================================================

class TradingEnv(gym.Env):
    def __init__(
        self,
        ticker="PETR4.SA",
        start="2019-01-01",
        end="2023-01-01",
        csv_path=None,
        window_size=20,
        lot_size=100,
        starting_cash=100_000.0,
        max_trade_fraction=0.1,
        transaction_fee_rate=0.0005,
        # Reward settings
        reward_mode="risk_adj",  # "log" or "risk_adj"
        risk_window=20,
        downside_only=False,
        dd_penalty=0.05,
        turnover_penalty=0.002, # base per unit turnover
        loss_penalty=0.10,
        inv_mom_penalty=0.02,   # penaliza posição sob momentum negativo (sma5<sma20)
        sell_turnover_factor=0.5,  # peso menor para penalizar vendas vs compras
    ):
        super().__init__()
        # Carregar dados: de cache (csv) se fornecido, senão baixar via yfinance
        if csv_path is not None and os.path.exists(csv_path):
            print(f"📁 Carregando dados de cache: {csv_path}")
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            print(f"📊 Baixando dados de {ticker} de {start} até {end}...")
            data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            raise ValueError("Sem dados retornados do yfinance. Verifique o ticker ou as datas.")

        # --- Extrair coluna de preço de forma robusta ---
        if isinstance(data.columns, pd.MultiIndex):
            flat_cols = ['_'.join(col).strip() for col in data.columns.values]
            data.columns = flat_cols

        candidatos = [
            'Adj Close', 'Adj_Close', 'Close',
            f'{ticker}_Adj Close', f'{ticker}_Adj_Close', f'{ticker}_Close',
            f'Adj Close_{ticker}', f'Adj_Close_{ticker}', f'Close_{ticker}'
        ]
        price_col = next((c for c in candidatos if c in data.columns), None)
        if price_col is None:
            raise ValueError(f"Nenhuma coluna de preço encontrada. Colunas disponíveis: {list(data.columns)}")

        # Guardar informações principais
        cleaned = data[price_col].copy()
        # garantir valores numéricos e suficientes pontos
        cleaned = pd.to_numeric(cleaned, errors='coerce').dropna()
        if len(cleaned) < max(2, window_size + 1):
            raise ValueError(
                f"Dados insuficientes após limpeza: {len(cleaned)} pontos; necessário >= {max(2, window_size + 1)}"
            )
        self.timestamps = cleaned.index  # timestamps reais do yfinance
        self.prices = cleaned.values
        # Tentar capturar OHLCV para indicadores adicionais
        def _find_col(df: pd.DataFrame, base: str) -> str | None:
            variants = [
                base, base.title(), base.upper(),
                f'{ticker}_{base}', f'{ticker}_{base.title()}', f'{ticker}_{base.upper()}',
                f'{base}_{ticker}', f'{base.title()}_{ticker}', f'{base.upper()}_{ticker}',
            ]
            for v in variants:
                if v in df.columns:
                    return v
            return None
        try:
            c_close = _find_col(data, 'Close') or price_col
            c_high = _find_col(data, 'High')
            c_low = _find_col(data, 'Low')
            c_vol = _find_col(data, 'Volume')
            self.close_prices = pd.to_numeric(data[c_close], errors='coerce').to_numpy() if c_close in data.columns else self.prices
            self.high = pd.to_numeric(data[c_high], errors='coerce').to_numpy() if c_high in data.columns else None
            self.low = pd.to_numeric(data[c_low], errors='coerce').to_numpy() if c_low in data.columns else None
            self.volume = pd.to_numeric(data[c_vol], errors='coerce').to_numpy() if c_vol in data.columns else None
        except Exception:
            self.close_prices = self.prices
            self.high = None
            self.low = None
            self.volume = None
        self.returns = np.diff(self.prices) / self.prices[:-1]
        # Parâmetros do ambiente
        self.window_size = max(20, int(window_size))  # garantir janela suficiente para SMA/RSI
        self.lot_size = int(lot_size)
        self.ticker = ticker
        # Guardar datas para rotular caches de indicadores
        self._start_label = str(start)
        self._end_label = str(end)
        self.starting_cash = float(starting_cash)
        self.max_trade_fraction = float(max_trade_fraction)
        self.transaction_fee_rate = float(transaction_fee_rate)
        self.max_steps = len(self.prices) - 1
        # Reward config
        self.reward_mode = reward_mode
        self.risk_window = max(5, int(risk_window))
        self.downside_only = bool(downside_only)
        self.dd_penalty = float(dd_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.loss_penalty = float(loss_penalty)
        self.inv_mom_penalty = float(inv_mom_penalty)
        self.sell_turnover_factor = float(sell_turnover_factor)

        # Espaços do ambiente
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        # Observation features (9):
        # stoch_diff, roll_std, atr14_norm, adx14, macd_hist, roc5, ret_z, rsi14, ema_ratio
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.trades = []
        # Pré-computar indicadores usados em observação e recompensa
        self._precompute_indicators()
        self.reset()

    def _indicator_cache_path(self) -> str:
        os.makedirs(os.path.join("data", "indicators"), exist_ok=True)
        safe_ticker = self.ticker.replace(":", "_").replace("/", "_")
        fname = f"{safe_ticker}_{self._start_label}_{self._end_label}_w{self.window_size}.npz"
        return os.path.join("data", "indicators", fname)

    def _precompute_indicators(self) -> None:
        ohlcv = Ohlcv(
            prices=self.prices,
            returns=self.returns,
            close=self.close_prices if self.close_prices is not None else self.prices,
            high=self.high,
            low=self.low,
            volume=self.volume,
        )
        feats = precompute_with_cache(
            ticker=self.ticker,
            start=self._start_label,
            end=self._end_label,
            ohlcv=ohlcv,
            window_size=self.window_size,
        )
        self.sma5 = feats["sma5"]
        self.sma20 = feats["sma20"]
        self.rsi14 = feats["rsi14"]
        self.ema12 = feats["ema12"]
        self.ema26 = feats["ema26"]
        self.macd_hist = feats["macd_hist"]
        self.roc5 = feats["roc5"]
        self.roc20 = feats["roc20"]
        self.roll_std = feats["roll_std"]
        self.atr14_norm = feats["atr14_norm"]
        self.stoch_diff = feats["stoch_diff"]
        self.adx14 = feats["adx14"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start after all feature lookbacks are available (MACD/EMA26/ADX need warmup)
        self.current_step = max(self.window_size, 35)
        self.balance = float(self.starting_cash)
        self.shares = 0
        self.prev_portfolio_value = float(self.starting_cash)
        self.trades = []
        # For reward calculations
        self._ret_hist = []
        self._eq_roll_max = self.prev_portfolio_value
        return self._get_obs(), {}

    def _get_obs(self):
        k = self.current_step
        # indicadores pré-computados alinhados em k
        rsi = float(self.rsi14[k])
        vol = float(self.roll_std[k]) if k < len(self.roll_std) else np.nan
        ema12 = float(self.ema12[k]) if hasattr(self, "ema12") else np.nan
        ema26 = float(self.ema26[k]) if hasattr(self, "ema26") else np.nan
        ema_ratio = (ema12 / (ema26 + 1e-8)) - 1.0
        macd_h = float(self.macd_hist[k]) if hasattr(self, "macd_hist") else np.nan
        roc5 = float(self.roc5[k]) if hasattr(self, "roc5") else np.nan
        atrn = float(self.atr14_norm[k]) if hasattr(self, "atr14_norm") else np.nan
        stochd = float(self.stoch_diff[k]) if hasattr(self, "stoch_diff") else np.nan
        adx = float(self.adx14[k]) if hasattr(self, "adx14") else np.nan
        ret_z = float(self.ret_z[k]) if hasattr(self, "ret_z") else np.nan

        # Ordem: stoch_diff, roll_std, atr14_norm, adx14, macd_hist, roc5, ret_z, rsi14, ema_ratio
        obs = np.array([
            stochd,
            vol,
            atrn,
            adx,
            macd_h,
            roc5,
            ret_z,
            rsi,
            ema_ratio,
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)



    def step(self, action):
        current_price = self.prices[self.current_step]
        # Timestamp real do candle atual
        timestamp = self.timestamps[self.current_step]
        self.current_step += 1
        next_price = self.prices[self.current_step]

        equity_before = self.balance + self.shares * current_price
        # taxas serão aplicadas apenas em operações de compra/venda, sobre o valor negociado
        fees = 0.0
        side = None
        qty = 0

        if action == 1:  # BUY em lotes, sem short
            # Cap estrito: até 5% do caixa disponível, sem mínimo forçado
            lot_value = self.lot_size * current_price
            affordable_lots = int(self.balance // lot_value)
            cap_lots = int((self.max_trade_fraction * self.balance) // lot_value)
            buy_lots = min(affordable_lots, cap_lots)
            if buy_lots > 0:
                qty = buy_lots * self.lot_size
                cost = qty * current_price
                trade_value = qty * current_price
                fees = self.transaction_fee_rate * trade_value
                self.balance -= cost + fees
                self.shares += qty
                side = "buy"

        elif action == 2 and self.shares > 0:  # SELL em lotes
            # Cap: até 5% da posição atual em lotes, com mínimo de 1 lote se houver posição
            position_lots = int(self.shares // self.lot_size)
            raw_cap = int(self.max_trade_fraction * position_lots)
            cap_lots = max(1, raw_cap) if position_lots > 0 else 0
            sell_lots = min(position_lots, cap_lots)
            qty = sell_lots * self.lot_size
            if qty > 0:
                revenue = qty * current_price
                trade_value = qty * current_price
                fees = self.transaction_fee_rate * trade_value
                self.balance += revenue - fees
                self.shares -= qty
                side = "sell"

        equity_after = self.balance + self.shares * next_price

        # Base log return of portfolio between steps
        r_t = float(np.log((equity_after + 1e-8) / (equity_before + 1e-8)))

        # Rolling risk (vol or downside vol)
        self._ret_hist.append(r_t)
        window = self._ret_hist[-self.risk_window :]
        if self.downside_only:
            negatives = [x for x in window if x < 0]
            sigma = np.std(negatives) if len(negatives) > 1 else np.std(window)
        else:
            sigma = np.std(window)
        sigma = float(sigma) + 1e-8

        # Drawdown using equity_after
        self._eq_roll_max = max(self._eq_roll_max, equity_after)
        dd = max(0.0, 1.0 - (equity_after / (self._eq_roll_max + 1e-8)))

        # Turnover from traded value
        traded_value = abs(qty) * current_price
        turnover = traded_value / (equity_before + 1e-8)

        # Momentum (sma5 vs sma20) and inventory fraction (usar pré-computados)
        sma5_k = float(self.sma5[self.current_step])
        sma20_k = float(self.sma20[self.current_step])
        mom_ratio = float((sma5_k / (sma20_k + 1e-8)) - 1.0)
        neg_mom = max(0.0, -mom_ratio)
        pos_frac = (self.shares * current_price) / (equity_before + 1e-8)

        # Risk-adjusted reward base
        #base = r_t / sigma if self.reward_mode == "risk_adj" else r_t
        # smooting Sortino try:
        ratio = r_t / (sigma + 1e-8)
        base = np.tanh(ratio) if self.reward_mode == "risk_adj" else r_t

        # Loss penalty: stronger penalty when return is negative
        loss_pen = self.loss_penalty * abs(r_t) if r_t < 0 else 0.0
        # Asymmetric turnover penalty: penalize buys fully, sells partially
        t_over = 0.0
        if side == "buy":
            t_over = self.turnover_penalty * turnover
        elif side == "sell":
            t_over = self.turnover_penalty * self.sell_turnover_factor * turnover
        # Inventory under negative momentum penalty
        inv_pen = self.inv_mom_penalty * pos_frac * neg_mom

        reward = base - (self.dd_penalty * dd) - t_over - loss_pen - inv_pen
        done = self.current_step >= self.max_steps - 1

        # só registra se houver trade
        trade_rec = None
        if side:
            trade_rec = {
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "symbol": self.ticker,
                "side": side,
                "qty": int(qty),
                "price": float(current_price),
                "fees": float(fees),
                "equity_before": float(equity_before),
                "equity_after": float(equity_after),
                "position_qty_after": int(self.shares),
            }
            self.trades.append(trade_rec)

        return self._get_obs(), reward, done, False, {
            "portfolio": equity_after,
            "r_t": r_t,
            "sigma": sigma,
            "dd": dd,
            "turnover": turnover,
            "traded": bool(side),
            "shares": int(self.shares),
            "trade": trade_rec,
        }


# ============================================================
#  Callback
# ============================================================

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0 and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            self.rewards.append(mean_reward)
            if self.verbose:
                print(f"Steps: {self.n_calls}, Mean Reward: {mean_reward:.3f}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate PPO trading agent with configurable windows")
    parser.add_argument("--window-size", "-w", type=int, default=20, help="Observation window size for features (default: 20)")
    parser.add_argument("--risk-window", "-r", type=int, default=20, help="Rolling risk window for reward adjustment (default: 20)")
    parser.add_argument("--train-start", type=str, default="2022-01-01",help="Training data start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, default="2023-01-01",help="Training data end date (YYYY-MM-DD)")
    parser.add_argument("--eval-start", type=str, default="2023-01-02",help="Evaluation data start date (YYYY-MM-DD)")
    parser.add_argument("--eval-end", type=str, default="2025-12-31", help="Evaluation data end date (YYYY-MM-DD)")
    parser.add_argument("--eval-slices", type=str, nargs='*', help=("Optional extra evaluation ranges as start:end pairs. Example: --eval-slices 2024-01-02:2024-12-31 2025-01-02:2025-10-23"),
    )
    parser.add_argument("--log-action-probs", type=int, default=0, help=("If > 0, logs per-step action probabilities for up to N steps of each evaluation slice to CSV."),
    )
    # Optional reward/behavior knobs
    parser.add_argument("--reward-mode", type=str, default="risk_adj", choices=["log", "risk_adj"], help="Base reward: log or risk_adj (r/sigma)")
    parser.add_argument("--dd-penalty", type=float, default=0.05, help="Drawdown penalty weight")
    parser.add_argument("--turnover-penalty", type=float, default=0.002, help="Turnover penalty weight")
    parser.add_argument("--loss-penalty", type=float, default=0.10, help="Negative return loss penalty weight")
    parser.add_argument("--inv-mom-penalty", type=float, default=0.02, help="Penalty for inventory under negative momentum")
    parser.add_argument("--sell-turnover-factor", type=float, default=0.5, help="Relative turnover penalty for sells vs buys (0–1)")
    parser.add_argument("--downside-only", action="store_true", help="Use downside-only volatility in risk-adjusted reward")
    parser.add_argument("--max-trade-fraction", type=float, default=0.1, help="Max fraction of cash/position per trade")
    parser.add_argument("--lot-size", type=int, default=100, help="Shares per lot for each trade")
    # PPO training knobs
    parser.add_argument("--total-timesteps", type=int, default=400_000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate")
    parser.add_argument("--batch-size", type=int, default=4096, help="PPO minibatch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per env before update")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient (exploration strength)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards (0-1)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter (0-1)")
    # Non-interactive mode
    parser.add_argument("--mode", choices=["train", "eval", "sweep", "retrain_best"], help="Run mode without prompt: train, eval, sweep or retrain_best")
    parser.add_argument("--net-arch", type=str, help="Comma-separated hidden sizes for policy MLP, e.g. 128,128")
    # Sweep knobs (stdout only)
    parser.add_argument("--sweep-trials", type=int, default=24, help="Number of random trials for sweep")
    parser.add_argument("--sweep-short-steps", type=int, default=120_000, help="Short training timesteps per trial")
    parser.add_argument("--sweep-seeds", type=int, default=2, help="Seeds averaged per trial")
    parser.add_argument("--sweep-min-improve", type=float, default=0.05, help="Early stop: min Sharpe improvement threshold")
    parser.add_argument("--sweep-patience", type=int, default=8, help="Early stop: number of trials without improvement")
    parser.add_argument("--sweep-exploit", action="store_true", help="Exploit mode: fix discrete knobs to current args and narrow LR around center")
    parser.add_argument("--sweep-center-lr", type=float, help="Center learning rate for exploit mode (if unset, uses --learning-rate)")
    parser.add_argument("--sweep-lr-span", type=float, default=0.3, help="Relative span around center LR for exploit (e.g., 0.3 => center*(1±0.3))")
    # Retrain from best sweep JSON
    parser.add_argument("--retrain-best-path", type=str, help="Path to sweep_best_*.json. If omitted, uses the latest in data/metrics/")
    # Reproducibility / performance knobs
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for training/eval")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel envs for training")
    parser.add_argument("--deterministic",action="store_true", help="Enable deterministic training (forces CPU, disables CuDNN benchmark, single env)")
    # Diagnostics / printing
    parser.add_argument("--print-params", action="store_true", help="Print all run parameters to terminal before executing")
    args = parser.parse_args()

    # Device selection (prefer GPU unless deterministic requested)
    auto_device = (
        "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    )
    device = "cpu" if args.deterministic else auto_device

    # Global seeding for reproducibility
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Optional deterministic guards (may reduce performance)
    if args.deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            import torch.backends.cudnn as cudnn  # type: ignore
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
        # Some CUDA ops require this env var for determinism
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Reduce thread-level nondeterminism
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
    print(f"Training device: {device}")
    print(
        f"Config: window_size={args.window_size}, risk_window={args.risk_window}, reward_mode={args.reward_mode}, "
        f"train=({args.train_start}→{args.train_end}), eval=({args.eval_start}→{args.eval_end})"
    )

    # Optional: print all run parameters used
    if getattr(args, "print_params", False):
        try:
            effective_num_envs = 1 if args.deterministic else max(1, int(args.num_envs))
        except Exception:
            effective_num_envs = args.num_envs
        params_out = {
            **{k: v for k, v in vars(args).items()},
            "device": device,
            "effective_num_envs": effective_num_envs,
            "ticker": "PETR4.SA",
        }
        print("\n=== Run Parameters ===")
        print(json.dumps(params_out, indent=2, sort_keys=True, ensure_ascii=False))
        print("======================\n")
    today = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Escolha do modo (non-interactive if --mode provided)
    mode = args.mode
    if mode is None:
        choice = input("\nEscolha o modo: [1] Treinar novo modelo  |  [2] Carregar e rodar existente: ")
        mode = "eval" if choice.strip() == "2" else "train"

    # Treinamento ou carregamento
    if mode == "eval" and os.path.exists("best_trading_model_PPO.zip"):
        print("\n" + "=" * 60)
        print("Carregando modelo salvo existente...")
        best_name = "PPO"
        # Carregar modelo diretamente (sem workers) e usar VecNormalize apenas na avaliação
        best_model = PPO.load("best_trading_model_PPO.zip", device=device)
        print("✓ Modelo carregado com sucesso!")
    elif mode == "train":
        print("\n" + "=" * 60)
        print("Treinando novos modelos PPO e A2C...")
        # Preparar cache de treino e um único env para validação do espaço
        train_csv = get_cached_data("PETR4.SA", args.train_start, args.train_end)
        _single_env = TradingEnv(
            ticker="PETR4.SA",
            start=args.train_start,
            end=args.train_end,
            csv_path=train_csv,
            window_size=args.window_size,
            reward_mode=args.reward_mode,
            lot_size=int(args.lot_size),
            max_trade_fraction=float(args.max_trade_fraction),
            risk_window=args.risk_window,
            downside_only=args.downside_only,
            dd_penalty=args.dd_penalty,
            turnover_penalty=args.turnover_penalty,
            loss_penalty=args.loss_penalty,
            inv_mom_penalty=args.inv_mom_penalty,
            sell_turnover_factor=args.sell_turnover_factor,
            starting_cash=100_000.0,
        )
        # Simple sanity check to avoid silent "no-buy" configurations
        try:
            p0 = float(_single_env.prices[_single_env.window_size])
            lot_value0 = int(args.lot_size) * p0
            cap_cash0 = float(args.max_trade_fraction) * float(_single_env.starting_cash)
            if lot_value0 > cap_cash0:
                print(
                    f"⚠️ Warning: lot_size×price0 ({lot_value0:.2f}) > max_trade_fraction×starting_cash ({cap_cash0:.2f}). "
                    "Buys may be impossible initially; consider reducing lot-size or increasing max-trade-fraction."
                )
        except Exception:
            pass
        try:
            _single_env.reset(seed=args.seed)
        except Exception:
            pass
        check_env(_single_env, warn=True)

        # Parallel environments apenas para treino
        def make_env():
            def _thunk():
                e = TradingEnv(
                    ticker="PETR4.SA",
                    start=args.train_start,
                    end=args.train_end,
                    csv_path=train_csv,
                    window_size=args.window_size,
                    reward_mode=args.reward_mode,
                    lot_size=int(args.lot_size),
                    max_trade_fraction=float(args.max_trade_fraction),
                    risk_window=args.risk_window,
                    downside_only=args.downside_only,
                    dd_penalty=args.dd_penalty,
                    turnover_penalty=args.turnover_penalty,
                    loss_penalty=args.loss_penalty,
                    inv_mom_penalty=args.inv_mom_penalty,
                    sell_turnover_factor=args.sell_turnover_factor,
                    starting_cash=100_000.0,
                )
                return Monitor(e)
            return _thunk

        n_envs = max(1, int(args.num_envs))
        if args.deterministic or n_envs == 1:
            vec_env = DummyVecEnv([make_env()])
        else:
            vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        try:
            vec_env.seed(args.seed)
        except Exception:
            pass
        print(f"Parallel envs: {n_envs}")
        # Net-arch parsing (train)
        net_arch = None
        if getattr(args, "net_arch", None):
            try:
                net_arch = [int(x.strip()) for x in str(args.net_arch).split(",") if x.strip()]
            except Exception:
                net_arch = None
        if not net_arch:
            net_arch = [256, 256]

        algorithms = {
            "PPO": PPO(
                "MlpPolicy",
                vec_env,
                device=device,
                verbose=0,
                learning_rate=float(args.learning_rate),
                n_steps=int(args.n_steps),  # per env; total = n_steps * n_envs
                batch_size=int(args.batch_size),
                n_epochs=int(args.n_epochs),
                policy_kwargs={"net_arch": net_arch},
                ent_coef=float(args.ent_coef),
                clip_range=float(args.clip_range),
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
                seed=args.seed,
            ),
        }
        results = {}
        for name, model in algorithms.items():
            print(f"\nTreinando {name}...")
            callback = ProgressCallback(check_freq=2000, verbose=0)
            model.learn(total_timesteps=int(args.total_timesteps), callback=callback, progress_bar=True)
            results[name] = {"model": model, "rewards": callback.rewards}
            print(f"✓ {name} concluído!")

        best_name = max(results.items(), key=lambda x: x[1]["model"].num_timesteps)[0]
        best_model = results[best_name]["model"]
        best_model.save(f"best_trading_model_{best_name}.zip")
        vec_env.save("vec_normalize.pkl")
        print(f"\n✓ Melhor modelo salvo: best_trading_model_{best_name}.zip")
    elif mode == "retrain_best":
        # Load best config JSON and retrain long with those params
        print("\n" + "=" * 60)
        print("Recarregando melhor configuração do sweep e treinando longo...")
        # Resolve path
        best_path = args.retrain_best_path
        if not best_path:
            try:
                dm = os.path.join("data", "metrics")
                candidates = [
                    os.path.join(dm, f) for f in os.listdir(dm)
                    if f.startswith("sweep_best_") and f.endswith(".json")
                ]
                if not candidates:
                    raise FileNotFoundError("Nenhum sweep_best_*.json encontrado em data/metrics/")
                best_path = max(candidates, key=lambda p: os.path.getmtime(p))
            except Exception as e:
                raise SystemExit(f"Falha ao localizar sweep_best JSON: {e}")
        with open(best_path, "r") as fh:
            best_blob = json.load(fh)
        cfg = best_blob.get("cfg", {})
        print(f"✓ Usando melhor configuração de {best_path}:\n{json.dumps(cfg, indent=2)}")

        # Prepare training env with cfg env knobs
        train_csv = get_cached_data("PETR4.SA", args.train_start, args.train_end)
        def make_env_retrain():
            def _thunk():
                e = TradingEnv(
                    ticker="PETR4.SA",
                    start=args.train_start,
                    end=args.train_end,
                    csv_path=train_csv,
                    window_size=args.window_size,
                    reward_mode=args.reward_mode,
                    lot_size=int(args.lot_size),
                    max_trade_fraction=float(args.max_trade_fraction),
                    risk_window=int(cfg.get("risk_window", args.risk_window)),
                    downside_only=bool(cfg.get("downside_only", args.downside_only)),
                    dd_penalty=float(cfg.get("dd_penalty", args.dd_penalty)),
                    turnover_penalty=0.0,
                    loss_penalty=args.loss_penalty,
                    inv_mom_penalty=float(cfg.get("inv_mom_penalty", getattr(args, "inv_mom_penalty", 0.0))),
                    sell_turnover_factor=args.sell_turnover_factor,
                    starting_cash=100_000.0,
                )
                return Monitor(e)
            return _thunk
        n_envs = max(1, int(args.num_envs))
        if args.deterministic or n_envs == 1:
            vec_env = DummyVecEnv([make_env_retrain()])
        else:
            vec_env = SubprocVecEnv([make_env_retrain() for _ in range(n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        # Net-arch: from cfg if present, else CLI, else default
        net_arch = cfg.get("net_arch")
        if getattr(args, "net_arch", None):
            try:
                net_arch = [int(x.strip()) for x in str(args.net_arch).split(",") if x.strip()]
            except Exception:
                pass
        if not net_arch:
            net_arch = [256, 256]

        model = PPO(
            "MlpPolicy",
            vec_env,
            device=device,
            verbose=0,
            learning_rate=float(cfg.get("learning_rate", args.learning_rate)),
            n_steps=int(cfg.get("n_steps", args.n_steps)),
            batch_size=int(cfg.get("batch_size", args.batch_size)),
            n_epochs=int(cfg.get("n_epochs", args.n_epochs)),
            policy_kwargs={"net_arch": net_arch},
            ent_coef=float(cfg.get("ent_coef", args.ent_coef)),
            clip_range=float(cfg.get("clip_range", args.clip_range)),
            gamma=float(cfg.get("gamma", args.gamma)),
            gae_lambda=float(cfg.get("gae_lambda", args.gae_lambda)),
            seed=args.seed,
        )
        total_steps = int(args.total_timesteps)
        print(f"Treinando por {total_steps} timesteps no dispositivo {device}...")
        model.learn(total_timesteps=total_steps, progress_bar=True)
        model.save(f"best_trading_model_PPO.zip")
        vec_env.save("vec_normalize.pkl")
        print("✓ Modelo reentreinado salvo como best_trading_model_PPO.zip e vec_normalize.pkl")
        # Prepare for downstream evaluation
        best_model = model
        best_name = "PPO"
    elif mode == "sweep":
        # Simple random sweep over PPO and env knobs, stdout only
        print("\n" + "=" * 60)
        print("Iniciando sweep de hiperparâmetros (stdout apenas)...")
        train_csv = get_cached_data("PETR4.SA", args.train_start, args.train_end)
        os.makedirs(os.path.join("data", "metrics"), exist_ok=True)
        sweep_csv = os.path.join("data", "metrics", f"sweep_results_{today}.csv")
        # Prepare CSV header if new file
        if not os.path.exists(sweep_csv):
            with open(sweep_csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "run_ts", "device", "trial", "mean_sharpe", "seeds", "seed_sharpes", "trial_wall_time_sec", "sweep_elapsed_sec",
                    "learning_rate", "n_steps", "batch_size", "n_epochs",
                    "ent_coef", "clip_range", "gamma", "gae_lambda",
                    "net_arch", "risk_window", "downside_only", "dd_penalty", "inv_mom_penalty"
                ])

        # Candidate grids
        LR = [1e-4, 3e-4]
        N_STEPS = [1024, 2048]
        BATCH = [4096, 8192]
        EPOCHS = [5, 10]
        ENT = [0.0, 0.005, 0.01]
        CLIP = [0.1, 0.2, 0.3]
        GAMMA = [0.95, 0.99]
        LAMBDA = [0.90, 0.95, 0.98]
        ARCH = [[128, 128], [256, 256]]
        RISK_WIN = [10, 20, 40]
        DOWNSIDE = [False, True]
        DD_PEN = [0.02, 0.05, 0.10]
        INV_MOM = [0.0, 0.01, 0.02]

        # Controlled sampling helpers
        def lhs_points(n: int, seed: int):
            rng = random.Random(seed)
            bins = list(range(n))
            rng.shuffle(bins)
            xs = []
            for b in bins:
                xs.append((b + rng.random()) / n)
            return xs

        def map_log_uniform(u: float, low: float, high: float) -> float:
            import math
            lo, hi = math.log(low), math.log(high)
            return float(math.exp(lo + u * (hi - lo)))

        lhs_lr = lhs_points(int(args.sweep_trials), seed=args.seed * 1237 + 17)

        def balanced_pick(lst, trial_idx: int, seed_off: int):
            rng = random.Random(args.seed * 97 + seed_off)
            perm = list(lst)
            rng.shuffle(perm)
            return perm[trial_idx % len(perm)]

        def sample_cfg(trial_idx: int):
            # Continuous: learning rate (explore vs exploit)
            if args.sweep_exploit:
                center = float(args.sweep_center_lr) if args.sweep_center_lr else float(args.learning_rate)
                span = float(args.sweep_lr_span)
                lo, hi = max(1e-6, center * (1 - span)), center * (1 + span)
                lr = map_log_uniform(lhs_lr[trial_idx - 1], lo, hi)
            else:
                lr = map_log_uniform(lhs_lr[trial_idx - 1], 1e-4, 3e-4)
            n_steps = balanced_pick(N_STEPS, trial_idx, 11)
            batch = balanced_pick(BATCH, trial_idx, 13)
            n_envs_eff = max(1, int(args.num_envs))
            if batch % n_envs_eff != 0 or batch > n_steps * n_envs_eff:
                for b in BATCH:
                    if b % n_envs_eff == 0 and b <= n_steps * n_envs_eff:
                        batch = b
                        break
            cfg = {
                "learning_rate": lr,
                "n_steps": n_steps if not args.sweep_exploit else int(args.n_steps),
                "batch_size": batch if not args.sweep_exploit else int(args.batch_size),
                "n_epochs": balanced_pick(EPOCHS, trial_idx, 17) if not args.sweep_exploit else int(args.n_epochs),
                "ent_coef": balanced_pick(ENT, trial_idx, 19),
                "clip_range": balanced_pick(CLIP, trial_idx, 23) if not args.sweep_exploit else float(args.clip_range),
                "gamma": balanced_pick(GAMMA, trial_idx, 29) if not args.sweep_exploit else float(args.gamma),
                "gae_lambda": balanced_pick(LAMBDA, trial_idx, 31) if not args.sweep_exploit else float(args.gae_lambda),
                "net_arch": balanced_pick(ARCH, trial_idx, 37),
                "risk_window": balanced_pick(RISK_WIN, trial_idx, 41) if not args.sweep_exploit else int(args.risk_window),
                "downside_only": balanced_pick(DOWNSIDE, trial_idx, 43) if not args.sweep_exploit else bool(args.downside_only),
                "dd_penalty": balanced_pick(DD_PEN, trial_idx, 47) if not args.sweep_exploit else float(args.dd_penalty),
                "inv_mom_penalty": random.choice(INV_MOM) if not args.sweep_exploit else (float(args.inv_mom_penalty) if hasattr(args, "inv_mom_penalty") else 0.01),
            }
            # Fix net_arch in exploit if provided
            if args.sweep_exploit and getattr(args, "net_arch", None):
                try:
                    cfg["net_arch"] = [int(x.strip()) for x in str(args.net_arch).split(",") if x.strip()]
                except Exception:
                    pass
            return cfg

        def make_env_slice(csv_path):
            def _thunk():
                e = TradingEnv(
                    ticker="PETR4.SA",
                    start=args.train_start,
                    end=args.train_end,
                    csv_path=csv_path,
                    window_size=args.window_size,
                    reward_mode=args.reward_mode,
                    lot_size=int(args.lot_size),
                    max_trade_fraction=float(args.max_trade_fraction),
                    risk_window=cfg["risk_window"],
                    downside_only=cfg["downside_only"],
                    dd_penalty=cfg["dd_penalty"],
                    turnover_penalty=0.0,
                    loss_penalty=args.loss_penalty,
                    inv_mom_penalty=cfg.get("inv_mom_penalty", 0.0),
                    sell_turnover_factor=args.sell_turnover_factor,
                    starting_cash=100_000.0,
                )
                return Monitor(e)
            return _thunk

        def eval_sharpe(model, vec_norm, start_date: str, end_date: str) -> float:
            test_csv = get_cached_data("PETR4.SA", start_date, end_date)
            eval_env = TradingEnv(
                ticker="PETR4.SA",
                start=start_date,
                end=end_date,
                csv_path=test_csv,
                window_size=args.window_size,
                reward_mode=args.reward_mode,
                lot_size=int(args.lot_size),
                max_trade_fraction=float(args.max_trade_fraction),
                risk_window=cfg["risk_window"],
                downside_only=cfg["downside_only"],
                dd_penalty=cfg["dd_penalty"],
                turnover_penalty=0.0,
                loss_penalty=args.loss_penalty,
                inv_mom_penalty=cfg.get("inv_mom_penalty", 0.0),
                sell_turnover_factor=args.sell_turnover_factor,
                starting_cash=100_000.0,
            )
            base_eval_env = Monitor(eval_env)
            from stable_baselines3.common.vec_env import DummyVecEnv
            eval_vec = DummyVecEnv([lambda: base_eval_env])
            from stable_baselines3.common.vec_env import VecNormalize
            eval_vec = VecNormalize(eval_vec, training=False, norm_obs=True, norm_reward=False)
            # Copy normalization stats from training
            try:
                eval_vec.obs_rms = vec_norm.obs_rms
            except Exception:
                pass
            obs = eval_vec.reset()
            done = False
            port_vals = [float(eval_env.starting_cash)]
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = eval_vec.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                port_vals.append(info.get("portfolio", port_vals[-1]))
                done = dones[0] if isinstance(dones, (list, tuple, np.ndarray)) else dones
            pv = np.array(port_vals, dtype=float)
            rets = np.diff(pv) / pv[:-1]
            std = np.std(rets)
            if std <= 1e-12:
                return float("nan")
            sharpe = (np.mean(rets) / std) * np.sqrt(252)
            return float(sharpe)

        best = None
        best_trial = None
        since_improve = 0
        sweep_t0 = time.monotonic()
        for t in range(1, int(args.sweep_trials) + 1):
            t0 = time.monotonic()
            cfg = sample_cfg(t)
            seed_list = [args.seed + i for i in range(int(args.sweep_seeds))]
            sharps = []
            for s in seed_list:
                # Seed
                os.environ["PYTHONHASHSEED"] = str(s)
                random.seed(s)
                np.random.seed(s)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)

                # Vec env for training
                n_envs = max(1, int(args.num_envs))
                if args.deterministic or n_envs == 1:
                    vec_env = DummyVecEnv([make_env_slice(train_csv)])
                else:
                    vec_env = SubprocVecEnv([make_env_slice(train_csv) for _ in range(n_envs)])
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    device=device,
                    verbose=0,
                    learning_rate=cfg["learning_rate"],
                    n_steps=cfg["n_steps"],
                    batch_size=cfg["batch_size"],
                    n_epochs=cfg["n_epochs"],
                    policy_kwargs={"net_arch": cfg["net_arch"]},
                    ent_coef=cfg["ent_coef"],
                    clip_range=cfg["clip_range"],
                    gamma=cfg["gamma"],
                    gae_lambda=cfg["gae_lambda"],
                    seed=s,
                )
                model.learn(total_timesteps=int(args.sweep_short_steps), progress_bar=False)
                # Evaluate on validation slice (args.eval_start/end)
                sh = eval_sharpe(model, vec_env, args.eval_start, args.eval_end)
                sharps.append(sh)
                try:
                    vec_env.close()
                except Exception:
                    pass
            # Seed-averaged Sharpe
            mean_sh = float(np.nanmean(sharps)) if len(sharps) else float("nan")
            cfg_print = {
                k: v for k, v in cfg.items() if k in [
                    "learning_rate","n_steps","batch_size","n_epochs","ent_coef","clip_range","gamma","gae_lambda","net_arch","risk_window","downside_only","dd_penalty"
                ]
            }
            trial_elapsed = time.monotonic() - t0
            sweep_elapsed = time.monotonic() - sweep_t0
            print(f"Trial {t:02d}: Sharpe(mean {args.sweep_seeds} seeds) = {mean_sh:+.3f} | time={trial_elapsed:.1f}s | cfg={cfg_print}")
            # Persist this trial to CSV
            try:
                seed_str = "|".join([f"s{i}:{sharps[i]:+.4f}" for i in range(len(sharps))])
                with open(sweep_csv, "a", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        today,
                        device,
                        t,
                        f"{mean_sh:+.6f}",
                        len(sharps),
                        seed_str,
                        f"{trial_elapsed:.3f}",
                        f"{sweep_elapsed:.3f}",
                        cfg_print.get("learning_rate"),
                        cfg_print.get("n_steps"),
                        cfg_print.get("batch_size"),
                        cfg_print.get("n_epochs"),
                        cfg_print.get("ent_coef"),
                        cfg_print.get("clip_range"),
                        cfg_print.get("gamma"),
                        cfg_print.get("gae_lambda"),
                        "-".join(map(str, cfg_print.get("net_arch", []))) if isinstance(cfg_print.get("net_arch"), (list, tuple)) else cfg_print.get("net_arch"),
                        cfg_print.get("risk_window"),
                        cfg_print.get("downside_only"),
                        cfg_print.get("dd_penalty"),
                        cfg.get("inv_mom_penalty"),
                    ])
            except Exception as e:
                print(f"⚠️ Falha ao persistir trial {t} no CSV: {e}")
            if (best is None) or (mean_sh > best + 1e-12):
                improvement = (float("-inf") if best is None else (mean_sh - best))
                best = mean_sh
                best_trial = {"trial": t, "sharpe": best, "cfg": cfg_print}
                since_improve = 0
            else:
                since_improve += 1
            if since_improve >= int(args.sweep_patience):
                # Early stop if no sufficient improvement
                if best_trial is not None:
                    print(f"Early stop: no improvement in {args.sweep_patience} trials. Best so far: {best_trial}")
                break
        # Persist best config
        if best_trial is not None:
            try:
                best_path = os.path.join("data", "metrics", f"sweep_best_{today}.json")
                with open(best_path, "w") as fh:
                    json.dump(best_trial, fh, indent=2)
                print(f"Melhor configuração salva em {best_path}")
            except Exception as e:
                print(f"⚠️ Falha ao salvar melhor configuração: {e}")
        # Exit after sweep
        return
    else:
        raise SystemExit("Modo 'eval' selecionado, mas o arquivo best_trading_model_PPO.zip não foi encontrado.")

    # Avaliação e métricas
    print("\n" + "=" * 60)
    print("Executando modelo e coletando métricas...")

    def _extract_price_series(csv_path: str, ticker: str):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        candidates = [
            'Adj Close', 'Adj_Close', 'Close',
            f'{ticker}_Adj Close', f'{ticker}_Adj_Close', f'{ticker}_Close',
            f'Adj Close_{ticker}', f'Adj_Close_{ticker}', f'Close_{ticker}'
        ]
        price_col = next((c for c in candidates if c in df.columns), None)
        if price_col is None:
            raise ValueError(f"Nenhuma coluna de preço encontrada em {csv_path}. Colunas: {list(df.columns)}")
        series = pd.to_numeric(df[price_col], errors='coerce').dropna()
        return series

    def run_eval_slice(start_date: str, end_date: str, label: str = None):
        slice_label = label or f"{start_date}_{end_date}"
        print("\n" + "=" * 60)
        print(f"Executando avaliação: {start_date} → {end_date} [{slice_label}]")

        test_csv = get_cached_data("PETR4.SA", start_date, end_date)
        eval_trading_env = TradingEnv(
            ticker="PETR4.SA",
            start=start_date,
            end=end_date,
            csv_path=test_csv,
            window_size=args.window_size,
            reward_mode=args.reward_mode,
            lot_size=int(args.lot_size),
            max_trade_fraction=float(args.max_trade_fraction),
            risk_window=args.risk_window,
            downside_only=args.downside_only,
            dd_penalty=args.dd_penalty,
            turnover_penalty=args.turnover_penalty,
            loss_penalty=args.loss_penalty,
            inv_mom_penalty=args.inv_mom_penalty,
            sell_turnover_factor=args.sell_turnover_factor,
            starting_cash=100_000.0,
        )
        base_eval_env = Monitor(eval_trading_env)
        eval_vec_env = DummyVecEnv([lambda: base_eval_env])
        if os.path.exists("vec_normalize.pkl"):
            eval_vec_env = VecNormalize.load("vec_normalize.pkl", eval_vec_env)
            eval_vec_env.training = False
            eval_vec_env.norm_reward = False
            print("✓ VecNormalize carregado para avaliação (norm_obs=True, norm_reward=False)")
        else:
            print("⚠️ VecNormalize não encontrado. Avaliação sem normalização de observações.")

        try:
            eval_vec_env.seed(args.seed)
        except Exception:
            pass
        obs = eval_vec_env.reset()
        try:
            starting_cash_eval = float(eval_trading_env.starting_cash)
        except Exception:
            starting_cash_eval = 1000.0
        portfolio_values = [starting_cash_eval]
        action_counts = {0: 0, 1: 0, 2: 0}
        exec_trades = 0
        collected_trades = []
        done = False
        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            try:
                a_int = int(action[0])
            except Exception:
                a_int = int(action)
            if a_int in action_counts:
                action_counts[a_int] += 1
            obs, rewards, dones, infos = eval_vec_env.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            portfolio_values.append(info.get("portfolio", portfolio_values[-1]))
            if info.get("traded"):
                exec_trades += 1
            tr = info.get("trade")
            if tr:
                collected_trades.append(tr)
            done = dones[0] if isinstance(dones, (list, tuple, np.ndarray)) else dones

        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.insert(returns, 0, 0)
        roll_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - roll_max) / roll_max

        ret_total = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        ret_mean = np.mean(returns) * 100
        ret_std = float(np.std(returns))
        vol_annual = ret_std * np.sqrt(252)
        sharpe = (np.mean(returns) / ret_std * np.sqrt(252)) if ret_std > 1e-12 else float('nan')
        max_dd = np.min(drawdown) * 100

        # Passive buy-and-hold baseline for the slice
        bh_series = _extract_price_series(test_csv, "PETR4.SA")
        bh_ret = ((bh_series.iloc[-1] / bh_series.iloc[0]) - 1.0) * 100.0
        print(f"Retorno buy-and-hold (slice): {bh_ret:8.2f}%")

        # Save per-slice metrics and trades with label to avoid overwrites
        metrics_df = pd.DataFrame({
            "portfolio": portfolio_values,
            "returns": returns,
            "drawdown": drawdown
        })
        metrics_df.to_csv(f"data/metrics/metrics_best_model_{slice_label}.csv", index=False)
        if slice_label == "main":
            # Preserve legacy path for main evaluation
            metrics_df.to_csv("data/metrics/metrics_best_model.csv", index=False)

        trades_df = pd.DataFrame(collected_trades)
        trades_df.to_csv(f"data/trades/trade_log_detailed_{today}_{slice_label}.csv", index=False)
        if slice_label == "main":
            trades_df.to_csv(f"data/trades/trade_log_detailed_{today}.csv", index=False)

        # Optional: log action probabilities for first N steps of the slice
        if args.log_action_probs and args.log_action_probs > 0:
            base_eval_env2 = Monitor(TradingEnv(
                ticker="PETR4.SA",
                start=start_date,
                end=end_date,
                csv_path=test_csv,
                window_size=args.window_size,
                reward_mode=args.reward_mode,
                lot_size=int(args.lot_size),
                max_trade_fraction=float(args.max_trade_fraction),
                risk_window=args.risk_window,
                downside_only=args.downside_only,
                dd_penalty=args.dd_penalty,
                turnover_penalty=args.turnover_penalty,
                loss_penalty=args.loss_penalty,
                inv_mom_penalty=args.inv_mom_penalty,
                sell_turnover_factor=args.sell_turnover_factor,
                starting_cash=100_000.0,
            ))
            log_env = DummyVecEnv([lambda: base_eval_env2])
            if os.path.exists("vec_normalize.pkl"):
                log_env = VecNormalize.load("vec_normalize.pkl", log_env)
                log_env.training = False
                log_env.norm_reward = False
            try:
                log_env.seed(args.seed)
            except Exception:
                pass
            log_obs = log_env.reset()
            rows = []
            steps_to_log = int(args.log_action_probs)
            for step_idx in range(max(1, steps_to_log)):
                with torch.no_grad():
                    log_obs_t = torch.as_tensor(log_obs, device=best_model.device)
                    dist = best_model.policy.get_distribution(log_obs_t)
                    probs = None
                    if hasattr(dist, 'distribution') and hasattr(dist.distribution, 'probs'):
                        probs = dist.distribution.probs[0].detach().cpu().numpy()
                act, _ = best_model.predict(log_obs, deterministic=True)
                log_obs, _, log_dones, _ = log_env.step(act)
                if probs is not None and len(probs) >= 3:
                    rows.append({
                        'step': step_idx,
                        'prob_hold': float(probs[0]),
                        'prob_buy': float(probs[1]),
                        'prob_sell': float(probs[2]),
                        'action': int(act[0]) if isinstance(act, (list, tuple, np.ndarray)) else int(act),
                    })
                if isinstance(log_dones, (list, tuple, np.ndarray)) and log_dones[0]:
                    break
            if rows:
                probs_df = pd.DataFrame(rows)
                outp = f"data/metrics/action_probs_{today}_{slice_label}.csv"
                probs_df.to_csv(outp, index=False)
                print(f"✓ Action probabilities salvos em {outp}")
        print("✓ Trades salvos em trade_log_detailed.csv")
        print(
            f"Trades coletados: {len(collected_trades)}  |  Trades no env (pós-reset): {len(eval_trading_env.trades)}"
        )
        print(
            f"Ações escolhidas: hold={action_counts.get(0,0)}, buy={action_counts.get(1,0)}, sell={action_counts.get(2,0)}"
        )
        print(f"Trades efetivados (durante avaliação): {exec_trades}")

        print("\n" + "=" * 60)
        print("📊 MÉTRICAS FINANCEIRAS DO MODELO")
        print(f"Retorno total:        {ret_total:8.2f}%")
        print(f"Retorno médio diário: {ret_mean:8.4f}%")
        print(f"Volatilidade anual:   {vol_annual*100:8.2f}%")
        print(f"Sharpe Ratio:         {sharpe:8.3f}")
        print(f"Máximo Drawdown:      {max_dd:8.2f}%")
        print("=" * 60)

        plt.figure()
        plt.plot(portfolio_values, color="green", linewidth=2)
        plt.axhline(portfolio_values[0], color="red", linestyle="--")
        plt.title(f"Desempenho {best_name} [{slice_label}]")
        plt.xlabel("Passos")
        plt.ylabel("Valor do portfólio ($)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"data/graphs/run_results_{today}_{slice_label}.png", dpi=150, bbox_inches="tight")
        if slice_label == "main":
            plt.savefig(f"data/graphs/run_results_{today}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("✓ Gráficos e métricas gerados com sucesso!")
        return {
            "ret_total": ret_total,
            "ret_mean": ret_mean,
            "vol_annual": vol_annual * 100,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "actions": action_counts,
            "trades": len(collected_trades),
        }

    # Run the primary eval range first
    _ = run_eval_slice(args.eval_start, args.eval_end, label="main")

    # Then run any extra slices if provided (e.g., 2024-only, 2025-only)
    if args.eval_slices:
        for token in args.eval_slices:
            try:
                start_date, end_date = token.split(":", 1)
            except ValueError:
                print(f"Formato inválido para --eval-slices: '{token}'. Use START:END.")
                continue
            run_eval_slice(start_date.strip(), end_date.strip())


if __name__ == "__main__":
    main()
