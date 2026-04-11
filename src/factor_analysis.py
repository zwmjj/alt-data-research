"""Factor research toolkit — Spearman IC, ICIR, quintile returns, and
factor correlation vs traditional momentum/value/quality factors.

Input contract:

  signal : pd.DataFrame  (index=month-end, columns=tickers, values=signal score)
  rets   : pd.DataFrame  (same index/columns, values=forward monthly returns)

Everything in this module operates on that aligned panel pair. Upstream
(nlp_signals / data_fetcher) is responsible for the alignment; downstream
(this module) only cares about the panel shape.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────
# 1. Information Coefficient
# ────────────────────────────────────────────────────────────
def compute_ic_series(
    signal: pd.DataFrame,
    forward_rets: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Cross-sectional rank correlation (Spearman) between signal_t
    and forward_returns_{t+1} at each rebalance date.

    Returns a pd.Series of ICs indexed by month-end. Rows with fewer
    than 10 valid assets are dropped.
    """
    common_dates = signal.index.intersection(forward_rets.index)
    common_cols = signal.columns.intersection(forward_rets.columns)
    sig = signal.loc[common_dates, common_cols]
    ret = forward_rets.loc[common_dates, common_cols]

    ics = {}
    for t in common_dates:
        s_row = sig.loc[t]
        r_row = ret.loc[t]
        mask = s_row.notna() & r_row.notna()
        if mask.sum() < 10:
            continue
        if method == "spearman":
            ics[t] = s_row[mask].corr(r_row[mask], method="spearman")
        else:
            ics[t] = s_row[mask].corr(r_row[mask], method="pearson")
    return pd.Series(ics, name=f"{method}_ic").sort_index()


def compute_ic_stats(ic_series: pd.Series) -> Dict[str, float]:
    """Summarize an IC time series: mean, std, ICIR, t-stat, hit rate."""
    ic = ic_series.dropna()
    if len(ic) < 6:
        return {"ic_mean": 0, "ic_std": 0, "icir": 0, "t_stat": 0,
                "hit_rate": 0, "n_obs": int(len(ic))}
    mean = float(ic.mean())
    std = float(ic.std())
    icir = mean / std * np.sqrt(12) if std > 0 else 0  # annualized
    t_stat = mean / (std / np.sqrt(len(ic))) if std > 0 else 0
    hit = float((ic > 0).mean())
    return {
        "ic_mean": round(mean, 4),
        "ic_std": round(std, 4),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "hit_rate": round(hit, 4),
        "n_obs": int(len(ic)),
    }


# ────────────────────────────────────────────────────────────
# 2. Quintile portfolio returns
# ────────────────────────────────────────────────────────────
def quintile_returns(
    signal: pd.DataFrame,
    forward_rets: pd.DataFrame,
    n_bins: int = 5,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Sort assets into N equal-sized bins by signal at each date,
    compute the equal-weighted forward return of each bin.

    Returns a DataFrame indexed by date with columns Q1..QN plus a
    `LS` (long-short, top minus bottom) column.
    """
    dates = signal.index.intersection(forward_rets.index)
    cols = signal.columns.intersection(forward_rets.columns)

    rows = []
    for t in dates:
        s = signal.loc[t, cols].dropna()
        r = forward_rets.loc[t, cols].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_assets:
            continue
        s = s.loc[common]
        r = r.loc[common]
        try:
            bins = pd.qcut(s, n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        row = {"date": t}
        for b in range(n_bins):
            mask = bins == b
            if mask.sum() == 0:
                row[f"Q{b+1}"] = np.nan
            else:
                row[f"Q{b+1}"] = float(r[mask].mean())
        row["LS"] = row.get(f"Q{n_bins}", 0) - row.get("Q1", 0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def quintile_summary(qrets: pd.DataFrame, periods_per_year: int = 12) -> Dict:
    """Annualized mean return, vol, Sharpe for each quintile + LS."""
    out = {}
    for col in qrets.columns:
        s = qrets[col].dropna()
        if len(s) < 2 or s.std() == 0:
            out[col] = {"mean_ann": 0, "vol_ann": 0, "sharpe": 0, "n": int(len(s))}
            continue
        out[col] = {
            "mean_ann": round(float(s.mean() * periods_per_year), 4),
            "vol_ann":  round(float(s.std() * np.sqrt(periods_per_year)), 4),
            "sharpe":   round(float(s.mean() / s.std() * np.sqrt(periods_per_year)), 4),
            "n":        int(len(s)),
        }
    return out


# ────────────────────────────────────────────────────────────
# 3. Traditional benchmark factors
# ────────────────────────────────────────────────────────────
def build_momentum_factor(rets: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """12-month trailing momentum, skipping the most recent month to
    avoid short-term reversal contamination (Jegadeesh-Titman).
    """
    cum = (1 + rets).rolling(lookback).apply(lambda x: np.prod(x) - 1, raw=True)
    return cum.shift(1)


def build_lowvol_factor(rets: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Negative of 12-month rolling std — low-vol preferred."""
    return (-rets.rolling(lookback).std()).shift(1)


def build_mom1_reversal(rets: pd.DataFrame) -> pd.DataFrame:
    """Negative of last month's return — short-term mean reversion."""
    return -rets.shift(1)


def factor_correlation_matrix(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-sectional average Spearman correlation between factor panels.

    For each date, compute Spearman between every pair of (factor_i, factor_j)
    across assets, then average the per-date correlations.
    """
    names = list(factors.keys())
    n = len(names)
    corr = np.zeros((n, n))

    # Intersect all panels on common dates/assets first
    common_dates = factors[names[0]].index
    common_cols = factors[names[0]].columns
    for f in factors.values():
        common_dates = common_dates.intersection(f.index)
        common_cols = common_cols.intersection(f.columns)

    for i in range(n):
        for j in range(i, n):
            fi = factors[names[i]].loc[common_dates, common_cols]
            fj = factors[names[j]].loc[common_dates, common_cols]
            pairwise = []
            for t in common_dates:
                si = fi.loc[t].dropna()
                sj = fj.loc[t].dropna()
                c = si.index.intersection(sj.index)
                if len(c) < 10:
                    continue
                pairwise.append(si.loc[c].corr(sj.loc[c], method="spearman"))
            avg = float(np.nanmean(pairwise)) if pairwise else 0.0
            corr[i, j] = corr[j, i] = avg

    return pd.DataFrame(corr, index=names, columns=names)


# ────────────────────────────────────────────────────────────
# 4. Panel construction helpers
# ────────────────────────────────────────────────────────────
def build_signal_panel(
    scored_filings: pd.DataFrame,
    score_col: str,
    start: str,
    end: str,
    hold_months: int = 12,
) -> pd.DataFrame:
    """Turn a list of per-filing scores into a month-end signal panel.

    For each ticker-filing, forward-fill the score for `hold_months`
    months starting from the filing's filed_date. If a new filing
    arrives before the hold period expires, the new score replaces
    the old one (most recent 10-K wins).
    """
    dates = pd.date_range(start, end, freq="ME")
    tickers = scored_filings["ticker"].unique()
    panel = pd.DataFrame(index=dates, columns=tickers, dtype=float)

    for ticker in tickers:
        ticker_df = (scored_filings[scored_filings["ticker"] == ticker]
                     .dropna(subset=["filed_date"])
                     .sort_values("filed_date"))
        for _, row in ticker_df.iterrows():
            filed = pd.Timestamp(row["filed_date"])
            start_mon = filed + pd.offsets.MonthEnd(0)
            end_mon = start_mon + pd.DateOffset(months=hold_months)
            window = (panel.index >= start_mon) & (panel.index < end_mon)
            panel.loc[window, ticker] = row[score_col]

    return panel
