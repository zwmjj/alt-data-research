#!/usr/bin/env python
"""Task 2 end-to-end runner — 13F institutional holdings factor.

Pipeline:
  1. Load filers (29 top managers) + parse all 672 13F-HR filings
  2. CUSIP→ticker map for S&P 100 via WRDS CRSP
  3. Build n_holders / value / shares quarterly panels
  4. Compute QoQ delta signals (4 candidates)
  5. IC analysis at 1m / 3m / 6m forward return horizons
  6. Quintile sorts + factor correlation vs Task 1 sentiment + traditional factors
  7. Generate plots + summary CSV
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import get_sp100_tickers, fetch_returns
from src.data_fetcher_13f import (
    get_filer_ciks, build_holdings_panel, get_cusip_ticker_map,
)
from src.factor_13f import (
    build_holdings_panels, quarterly_signals, expand_quarterly_to_monthly,
)
from src.factor_analysis import (
    compute_ic_series, compute_ic_stats,
    quintile_returns, quintile_summary,
    build_momentum_factor, build_lowvol_factor, build_mom1_reversal,
    factor_correlation_matrix,
)

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
PROCESSED = ROOT / "data" / "processed"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()

    # ── 1. Universe + returns ────────────────────────────
    print("[1/7] loading universe + returns ...")
    sp100 = get_sp100_tickers()
    returns = fetch_returns(sp100["ticker"].tolist(), start="2017-12-01", end="2025-06-30")
    print(f"      returns: {returns.shape}")

    # ── 2. 13F holdings panels ───────────────────────────
    print("[2/7] parsing 13F filings into long panel ...")
    filers = get_filer_ciks()
    long_panel = build_holdings_panel(filers)
    print(f"      long panel: {len(long_panel):,} rows, "
          f"{long_panel['filer_cik'].nunique()} filers, "
          f"{long_panel['period_end'].nunique()} periods")

    cusip_map = get_cusip_ticker_map(sp100["ticker"].tolist())
    n_panel, v_panel, s_panel = build_holdings_panels(long_panel, cusip_map)
    print(f"      panels: {n_panel.shape[0]} quarters x {n_panel.shape[1]} tickers")

    # ── 3. Build quarterly signals ───────────────────────
    print("[3/7] building QoQ delta signals ...")
    sigs = quarterly_signals(n_panel, v_panel, s_panel)
    for name, sig in sigs.items():
        nz = (sig.abs() > 0).sum().sum()
        print(f"      {name:20s} non-zero cells: {nz}")

    # ── 4. Expand to monthly + IC analysis ───────────────
    print("[4/7] computing IC at 1m / 3m / 6m forward horizons ...")
    monthly_idx = pd.date_range("2018-01-31", "2025-06-30", freq="ME")

    forward_1m = returns.shift(-1)
    forward_3m = ((1 + returns).rolling(3).apply(lambda x: np.prod(x), raw=True) - 1).shift(-3)
    forward_6m = ((1 + returns).rolling(6).apply(lambda x: np.prod(x), raw=True) - 1).shift(-6)

    horizon_results = {}
    for sig_name, sig_df in sigs.items():
        monthly = expand_quarterly_to_monthly(sig_df, monthly_idx)
        horizon_results[sig_name] = {}
        for h_name, fwd in [("1m", forward_1m), ("3m", forward_3m), ("6m", forward_6m)]:
            ic = compute_ic_series(monthly, fwd)
            horizon_results[sig_name][h_name] = compute_ic_stats(ic)

    # Print IC summary table
    print(f"\n      {'signal':<20} {'horizon':<6} {'IC':>8} {'ICIR':>8} {'t':>8} {'hit':>8} {'n':>5}")
    for sig_name, horizons in horizon_results.items():
        for h_name, stats in horizons.items():
            print(f"      {sig_name:<20} {h_name:<6} {stats['ic_mean']:>+8.4f} "
                  f"{stats['icir']:>+8.3f} {stats['t_stat']:>+8.2f} "
                  f"{stats['hit_rate']:>8.1%} {stats['n_obs']:>5d}")

    # ── 5. Pick best signal × horizon, do quintile + corr ──
    print("\n[5/7] selecting strongest (signal, horizon) by |ICIR| ...")
    best = max(
        ((s, h, horizon_results[s][h]) for s in horizon_results for h in horizon_results[s]),
        key=lambda x: abs(x[2]["icir"]),
    )
    best_sig_name, best_horizon, best_stats = best
    print(f"      WINNER: {best_sig_name} @ {best_horizon}  "
          f"ICIR={best_stats['icir']:+.3f} (|t|={abs(best_stats['t_stat']):.2f})")

    best_signal_monthly = expand_quarterly_to_monthly(sigs[best_sig_name], monthly_idx)
    best_fwd = {"1m": forward_1m, "3m": forward_3m, "6m": forward_6m}[best_horizon]

    qrets = quintile_returns(best_signal_monthly, best_fwd, n_bins=5, min_assets=20)
    qsummary = quintile_summary(qrets, periods_per_year=12)
    print(f"      quintile annual Sharpe at {best_horizon}:")
    for q, v in qsummary.items():
        print(f"        {q}: mean={v['mean_ann']:+.3f} vol={v['vol_ann']:.3f} SR={v['sharpe']:+.3f}")

    # ── 6. Factor correlation vs traditional factors ─────
    print("\n[6/7] factor correlation vs traditional factors ...")
    mom12 = build_momentum_factor(returns, lookback=12)
    mom1  = build_mom1_reversal(returns)
    lowvol = build_lowvol_factor(returns, lookback=12)
    # Also load Task 1 sentiment if available
    sec_level = None
    sf_path = PROCESSED / "filings_scored.parquet"
    if sf_path.exists():
        from src.factor_analysis import build_signal_panel
        scored = pd.read_parquet(sf_path)
        sec_level = -build_signal_panel(scored, "net_neg", start="2018-01-31", end="2025-06-30", hold_months=12)

    corr_inputs = {
        "13f_best": best_signal_monthly,
        "delta_n_holders": expand_quarterly_to_monthly(sigs["delta_n_holders"], monthly_idx),
        "mom12": mom12,
        "mom1_reversal": mom1,
        "lowvol": lowvol,
    }
    if sec_level is not None:
        corr_inputs["sec_sentiment_level"] = sec_level
    factor_corr = factor_correlation_matrix(corr_inputs)
    print(factor_corr.round(3).to_string())

    # ── 7. Plots + CSV ───────────────────────────────────
    print("\n[7/7] generating plots ...")
    _plot_horizon_decay(horizon_results, FIGS / "13f_horizon_decay.png")
    _plot_quintile_returns(qsummary, FIGS / "13f_quintile_returns.png", title=f"13F {best_sig_name} @ {best_horizon}")
    _plot_factor_corr(factor_corr, FIGS / "13f_factor_correlation.png")
    _plot_cumulative_ls(qrets, FIGS / "13f_cumulative_ls.png", title=f"13F L/S — {best_sig_name} @ {best_horizon}")
    _plot_n_holders_distribution(n_panel, FIGS / "13f_n_holders_dist.png")

    # Summary CSV
    summary = {
        "study": "task2_13f_institutional_holdings",
        "primary_signal": best_sig_name,
        "primary_horizon": best_horizon,
        "n_filers": int(long_panel["filer_cik"].nunique()),
        "n_filings": int(long_panel.groupby(["filer_cik", "period_end"]).ngroups),
        "n_quarters": int(n_panel.shape[0]),
        "n_holdings_rows": int(len(long_panel)),
        "n_universe_tickers": int(n_panel.shape[1]),
        **best_stats,
        "q1_sharpe": qsummary.get("Q1", {}).get("sharpe", 0),
        "q5_sharpe": qsummary.get("Q5", {}).get("sharpe", 0),
        "ls_sharpe": qsummary.get("LS", {}).get("sharpe", 0),
        "ls_mean_ann": qsummary.get("LS", {}).get("mean_ann", 0),
        "corr_vs_mom12": factor_corr.loc["13f_best", "mom12"],
        "corr_vs_mom1": factor_corr.loc["13f_best", "mom1_reversal"],
        "corr_vs_lowvol": factor_corr.loc["13f_best", "lowvol"],
    }
    if sec_level is not None:
        summary["corr_vs_sec_sentiment"] = factor_corr.loc["13f_best", "sec_sentiment_level"]

    pd.DataFrame([summary]).to_csv(RESULTS / "summary_stats_task2.csv", index=False)
    factor_corr.to_csv(RESULTS / "factor_correlation_task2.csv")

    # IC matrix CSV: rows=signals, columns=horizons
    ic_matrix_rows = []
    for sig_name, horizons in horizon_results.items():
        row = {"signal": sig_name}
        for h, stats in horizons.items():
            row[f"ic_{h}"] = stats["ic_mean"]
            row[f"icir_{h}"] = stats["icir"]
            row[f"t_{h}"] = stats["t_stat"]
        ic_matrix_rows.append(row)
    pd.DataFrame(ic_matrix_rows).to_csv(RESULTS / "ic_horizon_matrix_task2.csv", index=False)

    print(f"\n[done] elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"       results: {RESULTS}")
    return {
        "horizon_results": horizon_results,
        "best": best,
        "qsummary": qsummary,
        "factor_corr": factor_corr,
        "summary": summary,
    }


# ────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────
def _plot_horizon_decay(horizon_results, path):
    """Bar chart: ICIR by signal × horizon."""
    sig_names = list(horizon_results.keys())
    horizons = ["1m", "3m", "6m"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(len(sig_names))
    colors = ["#1a9641", "#fdae61", "#d7191c"]
    for i, h in enumerate(horizons):
        icirs = [horizon_results[s][h]["icir"] for s in sig_names]
        ax.bar(x + i*width, icirs, width, label=f"forward {h}", color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(sig_names, rotation=20, ha="right")
    ax.set_ylabel("ICIR (annualized)")
    ax.set_title("13F Holdings Factors — IC Information Ratio by Forward Horizon")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_quintile_returns(qsummary, path, title):
    qs = sorted([k for k in qsummary if k.startswith("Q")])
    means = [qsummary[k]["mean_ann"] for k in qs]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(qs, means, color=["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{m:+.2%}", ha="center",
                va="bottom" if m > 0 else "top", fontsize=9)
    ls = qsummary.get("LS", {}).get("mean_ann", 0)
    ax.set_title(f"{title}  |  L/S = {ls:+.2%}")
    ax.set_ylabel("Annualized mean return")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_factor_corr(corr, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    ax.set_title("Cross-Sectional Factor Correlation\n13F + Task 1 sentiment + traditional")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_cumulative_ls(qrets, path, title):
    ls = qrets["LS"].dropna()
    if len(ls) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    cum = (1 + ls).cumprod()
    ax.plot(cum.index, cum.values, color="darkgreen", linewidth=1.5, label="L/S cumulative")
    ax.axhline(1.0, color="k", linewidth=0.5)
    total = cum.iloc[-1] - 1
    ax.set_title(f"{title}  ({total:+.1%} total, n={len(ls)} months)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_n_holders_distribution(n_panel, path):
    """How concentrated is institutional ownership across our universe?"""
    last = n_panel.iloc[-1].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(last)), last.values, color="steelblue", alpha=0.85)
    ax.set_title(f"S&P 100 stocks ranked by # of top-tier 13F holders ({n_panel.index[-1].date()})")
    ax.set_xlabel("ticker rank")
    ax.set_ylabel(f"# of top-tier filers holding (max = {len(get_filer_ciks())})")
    ax.grid(alpha=0.3, axis="y")
    # Annotate top 5
    for i in range(min(5, len(last))):
        ax.text(i, last.iloc[i] + 0.2, last.index[i], ha="center", fontsize=8, rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# Re-import inside helper to avoid circular issue
def get_filer_ciks():
    from src.data_fetcher_13f import get_filer_ciks as _g
    return _g()


if __name__ == "__main__":
    main()
