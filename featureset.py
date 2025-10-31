import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import talib


@dataclass
class Ohlcv:
    prices: np.ndarray
    returns: np.ndarray
    close: np.ndarray
    high: Optional[np.ndarray]
    low: Optional[np.ndarray]
    volume: Optional[np.ndarray]


def indicator_cache_path(ticker: str, start: str, end: str, window_size: int) -> str:
    os.makedirs(os.path.join("data", "indicators"), exist_ok=True)
    safe = ticker.replace(":", "_").replace("/", "_")
    fname = f"{safe}_{start}_{end}_w{window_size}.npz"
    return os.path.join("data", "indicators", fname)


def compute_indicators(ohlcv: Ohlcv, window_size: int) -> Dict[str, np.ndarray]:
    p = ohlcv.prices.astype(float)
    # Core
    sma5 = talib.SMA(p, timeperiod=5)
    sma20 = talib.SMA(p, timeperiod=20)
    rsi14 = talib.RSI(p, timeperiod=14)
    ema12 = talib.EMA(p, timeperiod=12)
    ema26 = talib.EMA(p, timeperiod=26)
    _macd, _macd_sig, macd_hist = talib.MACD(p, fastperiod=12, slowperiod=26, signalperiod=9)
    roc5 = talib.ROC(p, timeperiod=5)
    roc20 = talib.ROC(p, timeperiod=20)
    # Rolling std over returns aligned to price index length
    ret_s = pd.Series(p).pct_change()
    roll_std = ret_s.rolling(window=window_size, min_periods=window_size).std().to_numpy()
    # Return z-score over window_size
    mu = ret_s.rolling(window=window_size, min_periods=window_size).mean()
    sd = ret_s.rolling(window=window_size, min_periods=window_size).std()
    ret_z = ((ret_s - mu) / (sd + 1e-8)).to_numpy()

    # OHLC-dependent
    if ohlcv.high is not None and ohlcv.low is not None and ohlcv.close is not None:
        atr14 = talib.ATR(ohlcv.high.astype(float), ohlcv.low.astype(float), ohlcv.close.astype(float), timeperiod=14)
        atr14_norm = atr14 / (ohlcv.close + 1e-8)
        try:
            k, d = talib.STOCH(
                ohlcv.high.astype(float),
                ohlcv.low.astype(float),
                ohlcv.close.astype(float),
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            )
            stoch_diff = (k - d)
        except Exception:
            stoch_diff = np.full_like(p, np.nan, dtype=float)
        # ADX
        try:
            adx14 = talib.ADX(ohlcv.high.astype(float), ohlcv.low.astype(float), ohlcv.close.astype(float), timeperiod=14)
        except Exception:
            adx14 = np.full_like(p, np.nan, dtype=float)
    else:
        atr14_norm = np.full_like(p, np.nan, dtype=float)
        stoch_diff = np.full_like(p, np.nan, dtype=float)
        adx14 = np.full_like(p, np.nan, dtype=float)

    # Interaction feature: stoch_diff * adx14 (scale to 0-1 for ADX)
    with np.errstate(invalid='ignore'):
        stoch_adx = (stoch_diff / 100.0) * (adx14 / 100.0)

    return {
        "sma5": sma5,
        "sma20": sma20,
        "rsi14": rsi14,
        "ema12": ema12,
        "ema26": ema26,
        "macd_hist": macd_hist,
        "roc5": roc5,
        "roc20": roc20,
        "roll_std": roll_std,
        "ret_z": ret_z,
        "atr14_norm": atr14_norm,
        "stoch_diff": stoch_diff,
        "stoch_adx": stoch_adx,
        "adx14": adx14,
    }


def precompute_with_cache(
    ticker: str,
    start: str,
    end: str,
    ohlcv: Ohlcv,
    window_size: int,
) -> Dict[str, np.ndarray]:
    path = indicator_cache_path(ticker, start, end, window_size)
    needed = {
        "sma5",
        "sma20",
        "rsi14",
        "ema12",
        "ema26",
        "macd_hist",
        "roc5",
        "roc20",  
        "roll_std",
        "ret_z",
        "atr14_norm",
        "stoch_diff",
        "stoch_adx",
        "adx14",
    }
    if os.path.exists(path):
        try:
            data = np.load(path)
            if set(data.files) >= needed:
                return {k: data[k] for k in needed}
        except Exception:
            pass
    feats = compute_indicators(ohlcv, window_size)
    try:
        np.savez_compressed(path, **feats)
    except Exception:
        pass
    return feats


def extract_ohlcv(df: pd.DataFrame, ticker: str, price_col: str) -> Ohlcv:
    def find_col(base: str) -> Optional[str]:
        candidates = [
            base, base.title(), base.upper(),
            f"{ticker}_{base}", f"{ticker}_{base.title()}", f"{ticker}_{base.upper()}",
            f"{base}_{ticker}", f"{base.title()}_{ticker}", f"{base.upper()}_{ticker}",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    close_name = find_col("Close") or price_col
    high_name = find_col("High")
    low_name = find_col("Low")
    vol_name = find_col("Volume")

    close = pd.to_numeric(df[close_name], errors="coerce").to_numpy() if close_name in df.columns else None
    high = pd.to_numeric(df[high_name], errors="coerce").to_numpy() if high_name in df.columns else None
    low = pd.to_numeric(df[low_name], errors="coerce").to_numpy() if low_name in df.columns else None
    vol = pd.to_numeric(df[vol_name], errors="coerce").to_numpy() if vol_name in df.columns else None

    prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy()
    prices = prices.astype(float)
    returns = np.diff(prices) / prices[:-1]
    return Ohlcv(prices=prices, returns=returns, close=close, high=high, low=low, volume=vol)


def compute_feature_frame(
    df: pd.DataFrame, ticker: str, window_size: int, price_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join(col).strip() for col in df.columns.values]
    if price_col is None:
        candidates = [
            "Adj Close", "Adj_Close", "Close",
            f"{ticker}_Adj Close", f"{ticker}_Adj_Close", f"{ticker}_Close",
            f"Adj Close_{ticker}", f"Adj_Close_{ticker}", f"Close_{ticker}",
        ]
        price_col = next((c for c in candidates if c in df.columns), None)
        if price_col is None:
            raise ValueError(f"Nenhuma coluna de preço encontrada. Colunas: {list(df.columns)}")
    ohlcv = extract_ohlcv(df, ticker, price_col)
    feats = compute_indicators(ohlcv, window_size)
    prices = pd.to_numeric(df[price_col], errors="coerce").astype(float)
    rel_price_w = (prices / (prices.shift(window_size) + 1e-8)) - 1.0
    # Build feature frame aligned to index
    idx = prices.index
    frame = pd.DataFrame({
        "rel_price_w": rel_price_w.values,
        "sma_ratio": (feats["sma5"] / (feats["sma20"] + 1e-8)) - 1.0,
        "rsi14": feats["rsi14"],
        "roll_std": feats["roll_std"],
        "ret_z": feats["ret_z"],
        "ema_ratio": (feats["ema12"] / (feats["ema26"] + 1e-8)) - 1.0,
        "macd_hist": feats["macd_hist"],
        "roc5": feats["roc5"],
        "roc20": feats["roc20"],
        "atr14_norm": feats["atr14_norm"],
        "stoch_diff": feats["stoch_diff"],
        "stoch_adx": feats["stoch_adx"],
        "adx14": feats["adx14"],
    }, index=idx)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    fwd_ret = prices.pct_change().shift(-1).reindex(frame.index)
    frame = frame.loc[~fwd_ret.isna()].copy()
    fwd_ret = fwd_ret.loc[frame.index]
    return frame, fwd_ret
