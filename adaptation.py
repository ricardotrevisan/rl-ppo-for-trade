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
import torch
import os
import random
import warnings
from datetime import datetime, timedelta
import argparse
import json
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
        self.returns = np.diff(self.prices) / self.prices[:-1]
        # Parâmetros do ambiente
        self.window_size = max(20, int(window_size))  # garantir janela suficiente para SMA/RSI
        self.lot_size = int(lot_size)
        self.ticker = ticker
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.trades = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = float(self.starting_cash)
        self.shares = 0
        self.prev_portfolio_value = float(self.starting_cash)
        self.trades = []
        # For reward calculations
        self._ret_hist = []
        self._eq_roll_max = self.prev_portfolio_value
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step
        prices = self.prices[start:end+1]
        returns = self.returns[start:end]

        sma5 = talib.SMA(prices, timeperiod=5)[-1]
        sma20 = talib.SMA(prices, timeperiod=20)[-1]
        rsi = talib.RSI(prices, timeperiod=14)[-1]
        vol = np.std(returns)

        # Sem escalas manuais: deixar VecNormalize aprender os scalers
        obs = np.array([
            self.balance,
            float(self.shares),
            prices[-1] / prices[0] - 1.0,
            (sma5 / sma20) - 1.0,
            rsi,  # em pontos 0-100
            vol
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

        # Momentum (sma5 vs sma20) and inventory fraction
        w_start = max(0, self.current_step - self.window_size)
        w_end = self.current_step
        w_prices = self.prices[w_start:w_end+1]
        sma5 = talib.SMA(w_prices, timeperiod=5)[-1]
        sma20 = talib.SMA(w_prices, timeperiod=20)[-1]
        mom_ratio = float((sma5 / (sma20 + 1e-8)) - 1.0)
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
    parser.add_argument("--mode", choices=["train", "eval"], help="Run mode without prompt: train or eval")
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
                policy_kwargs={"net_arch": [256, 256]},
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
