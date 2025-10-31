import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import talib
import yfinance as yf
from featureset import compute_feature_frame


def get_cached_data(ticker: str, start: str, end: str, cache_dir: str = "data") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace(":", "_").replace("/", "_")
    fname = f"{safe_ticker}_{start}_{end}.csv"
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        print(f"Cache not found. Downloading {ticker} ({start}→{end}) once...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("No data from yfinance for cache.")
        df.to_csv(path)
        print(f"Saved cache: {path}")
    else:
        print(f"Using cache: {path}")
    return path


"""
Centralized features come from featureset.compute_feature_frame to stay consistent
with the RL environment. This script only prints to stdout.
"""


def high_corr_pairs(df: pd.DataFrame, thresh: float) -> List[Tuple[str, str, float]]:
    corr = df.corr(method="pearson")
    pairs = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) >= thresh:
                pairs.append((cols[i], cols[j], float(c)))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


def info_coefficient(feats: pd.DataFrame, target: pd.Series, splits: int = 2) -> pd.DataFrame:
    # Overall IC (Spearman)
    overall = feats.corrwith(target, method="spearman").rename("IC_overall")
    # Split ICs by equal-sized chunks
    n = len(feats)
    chunk = max(1, n // splits)
    split_cols = []
    for s in range(splits):
        start = s * chunk
        end = n if s == splits - 1 else (s + 1) * chunk
        sub = feats.iloc[start:end]
        tgt = target.iloc[start:end]
        split_ic = sub.corrwith(tgt, method="spearman").rename(f"IC_split{s+1}")
        split_cols.append(split_ic)
    out = pd.concat([overall] + split_cols, axis=1)
    # Stability metrics
    out["IC_abs_mean"] = out.filter(like="IC_").abs().mean(axis=1)
    signs = np.sign(out.filter(like="IC_")).replace(0, np.nan)
    out["sign_consistency"] = signs.mean(axis=1).abs()
    return out.sort_values(["IC_abs_mean", "sign_consistency"], ascending=False)


def greedy_decorrelation(feats: pd.DataFrame, ranking: List[str], corr_thresh: float) -> List[str]:
    kept: List[str] = []
    for f in ranking:
        if not kept:
            kept.append(f)
            continue
        cmax = max(abs(feats[f].corr(feats[k], method="pearson")) for k in kept)
        if cmax < corr_thresh:
            kept.append(f)
    return kept


def pca_analysis(feats: pd.DataFrame, top_components: int = 3, top_features: int = 5):
    """Run PCA on standardized features (no external deps) and print a summary.
    - Standardizes columns (z-score). Drops zero-variance columns.
    - Uses eigen-decomposition of the correlation matrix for stability.
    - Prints explained variance ratios and top-loading features per component.
    """
    X = feats.copy()
    cols = list(X.columns)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    nonzero = sd > 1e-12
    if not bool(nonzero.all()):
        dropped = [c for c, ok in zip(cols, nonzero) if not ok]
        if dropped:
            print("\nPCA: Dropping zero-variance features:")
            for d in dropped:
                print(f"  - {d}")
        X = X.loc[:, nonzero]
        cols = list(X.columns)
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)

    if X.shape[1] == 0:
        print("\nPCA: No valid features after dropping zero-variance; skipping.")
        return

    Z = (X - mu) / (sd + 1e-8)
    n = Z.shape[0]
    # Correlation matrix (since Z is standardized):
    # Using covariance of Z equals correlation; eigvals sum to n_features
    C = (Z.values.T @ Z.values) / max(1, (n - 1))
    # Eigen-decomposition; eigenvalues are variances explained per component
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total_var = eigvals.sum() if eigvals.sum() > 0 else 1.0
    evr = eigvals / total_var
    cum = np.cumsum(evr)

    print("\n=== PCA (standardized features) ===")
    print(f"Features used: {len(cols)}  Samples: {n}")
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    show = min(len(cols), max(top_components, 3))
    print("Explained variance ratio (first components):")
    for i in range(show):
        print(f"  PC{i+1}: {evr[i]*100:6.2f}%  (cumulative {cum[i]*100:6.2f}%)")
    print(f"Components to reach 90%: {k90} | 95%: {k95}")

    # Top loadings per component
    print("Top feature loadings (absolute) per component:")
    for i in range(min(show, top_components)):
        vec = eigvecs[:, i]
        pairs = list(zip(cols, vec))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        tops = pairs[:top_features]
        tops_str = ", ".join([f"{name} ({weight:+.2f})" for name, weight in tops])
        print(f"  PC{i+1}: {tops_str}")


def main():
    ap = argparse.ArgumentParser(description="Offline feature screening and selection (stdout only)")
    ap.add_argument("--ticker", default="PETR4.SA")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-01-01")
    ap.add_argument("--window-size", type=int, default=20)
    ap.add_argument("--corr-thresh", type=float, default=0.9)
    ap.add_argument("--splits", type=int, default=2)
    args = ap.parse_args()

    csv = get_cached_data(args.ticker, args.start, args.end)
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    feats, fwd_ret = compute_feature_frame(df, args.ticker, args.window_size)

    print("\n=== Feature Summary ===")
    print(f"Ticker: {args.ticker}  Range: {args.start} → {args.end}  N={len(feats)}")
    null_rates = feats.isna().mean().sort_values()
    print("Non-null coverage per feature:")
    for k, v in (1 - null_rates).items():
        print(f"  - {k:12s}: {v*100:6.2f}%")

    print("\n=== High Correlation Pairs (|ρ| ≥ {:.2f}) ===".format(args.corr_thresh))
    pairs = high_corr_pairs(feats, args.corr_thresh)
    if not pairs:
        print("  None")
    else:
        for a, b, c in pairs[:20]:
            print(f"  - {a:12s} ~ {b:12s}: ρ={c: .3f}")

    print("\n=== Information Coefficient (Spearman with next return) ===")
    ic = info_coefficient(feats, fwd_ret, splits=args.splits)
    cols_to_show = ["IC_overall"] + [c for c in ic.columns if c.startswith("IC_split")] + ["IC_abs_mean", "sign_consistency"]
    for name, row in ic[cols_to_show].iterrows():
        vals = [row[c] for c in cols_to_show]
        parts = "  ".join(f"{v: .3f}" if pd.notna(v) else "  nan" for v in vals)
        print(f"  - {name:12s}: {parts}")

    print("\n=== Suggested De-correlated Subset ===")
    ranking = list(ic.index)
    kept = greedy_decorrelation(feats, ranking, corr_thresh=args.corr_thresh)
    # Cap to a compact set (8–12)
    suggested = kept[:12]
    print("Keep (order by IC/stability, filtered by corr):")
    for f in suggested:
        print(f"  - {f}")

    print("\nNotes:")
    print("- Use this subset to freeze the observation features in RL.")
    print("- Recompute only on the training window; do not peek at validation/test.")
    print("- Prefer 1 trend (sma_ratio or ema_ratio), 1–2 momentum (roc5/roc20),")
    print("  1 oscillator (macd_hist or stoch_diff), and 1–2 volatility (roll_std, atr14_norm).")

    # PCA summary (optional, informative)
    pca_analysis(feats)


if __name__ == "__main__":
    main()
