#!/usr/bin/env python
"""Task 1 end-to-end runner.

Pipeline:
  1. Load S&P 100 tickers + returns
  2. Scan downloaded 10-K filings → LM sentiment scores per filing
  3. Build a monthly signal panel (forward-fill each 10-K's score for
     12 months after filing)
  4. Compute IC / ICIR vs forward monthly returns
  5. Quintile portfolio returns
  6. Compare correlation with traditional factors (mom12 / mom1 / lowvol)
  7. Generate plots + summary_stats.csv
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

from src.data_fetcher import get_sp100_tickers, list_10k_paths, fetch_returns
from src.nlp_signals import score_filings_from_manifest, enrich_filed_dates
from src.factor_analysis import (
    compute_ic_series, compute_ic_stats,
    quintile_returns, quintile_summary,
    build_momentum_factor, build_lowvol_factor, build_mom1_reversal,
    build_signal_panel, factor_correlation_matrix,
)

RESULTS_DIR = ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figures"
PROCESSED = ROOT / "data" / "processed"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()

    # ── 1. Universe + returns ────────────────────────────
    print("[1/7] loading S&P 100 tickers ...")
    sp100 = get_sp100_tickers()
    tickers = sp100["ticker"].tolist()

    print("[2/7] loading monthly returns (WRDS primary, yfinance fallback) ...")
    returns = fetch_returns(tickers, start="2017-01-01", end="2025-06-30")
    # Forward-shift returns so signal_t predicts return_{t+1}
    forward_rets = returns.shift(-1)

    # ── 2. Score all downloaded 10-Ks ────────────────────
    print("[3/7] scanning downloaded 10-K filings ...")
    manifest = list_10k_paths()
    print(f"      {len(manifest)} filings across {manifest['ticker'].nunique()} tickers")

    print("[4/7] scoring filings with Loughran-McDonald dictionary ...")
    scored = score_filings_from_manifest(manifest)
    # Enrich with precise filed dates from EDGAR JSON
    print("      enriching filed dates from EDGAR submissions API ...")
    scored = enrich_filed_dates(scored, sp100)
    scored.to_parquet(PROCESSED / "filings_scored.parquet")
    print(f"      saved to {PROCESSED / 'filings_scored.parquet'}")

    # ── 3. Build monthly signal panels ───────────────────
    print("[5/7] building monthly signal panels ...")
    # Two candidate sentiment factors to compare:
    #   sec_level  — level of net_neg (latest 10-K held 12 months forward)
    #   sec_change — year-over-year change in net_neg per ticker
    #
    # Both are flipped so higher-score = bullish (LM 2011: negative
    # sentiment predicts negative returns; we invert for readability).
    net_neg_level = build_signal_panel(
        scored, "net_neg", start="2018-01-31", end="2025-06-30", hold_months=12,
    )
    signal_level = -net_neg_level

    scored_sorted = scored.sort_values(["ticker", "filed_date"])
    scored_sorted["net_neg_change"] = scored_sorted.groupby("ticker")["net_neg"].diff()
    net_neg_change = build_signal_panel(
        scored_sorted.dropna(subset=["net_neg_change"]),
        "net_neg_change", start="2018-01-31", end="2025-06-30", hold_months=12,
    )
    signal_change = -net_neg_change  # rising negativity -> bearish -> flip

    # ── 4. IC analysis ───────────────────────────────────
    print("[6/7] computing IC, quintile returns, factor correlations ...")

    ic_level = compute_ic_series(signal_level, forward_rets, method="spearman")
    ic_level_stats = compute_ic_stats(ic_level)
    ic_change = compute_ic_series(signal_change, forward_rets, method="spearman")
    ic_change_stats = compute_ic_stats(ic_change)

    print(f"      LEVEL:  IC mean={ic_level_stats['ic_mean']:+.4f} "
          f"ICIR={ic_level_stats['icir']:+.3f} "
          f"t={ic_level_stats['t_stat']:+.2f} hit={ic_level_stats['hit_rate']:.1%} n={ic_level_stats['n_obs']}")
    print(f"      CHANGE: IC mean={ic_change_stats['ic_mean']:+.4f} "
          f"ICIR={ic_change_stats['icir']:+.3f} "
          f"t={ic_change_stats['t_stat']:+.2f} hit={ic_change_stats['hit_rate']:.1%} n={ic_change_stats['n_obs']}")

    # Pick the stronger of the two as primary for quintile / corr analysis.
    if abs(ic_change_stats["ic_mean"]) > abs(ic_level_stats["ic_mean"]):
        primary_signal = signal_change
        primary_label = "sec_sentiment_change"
        primary_row = "sec_change"
        ic = ic_change
        ic_stats = ic_change_stats
        print(f"      primary factor = sentiment CHANGE (stronger IC)")
    else:
        primary_signal = signal_level
        primary_label = "sec_sentiment_level"
        primary_row = "sec_level"
        ic = ic_level
        ic_stats = ic_level_stats
        print(f"      primary factor = sentiment LEVEL (stronger IC)")

    # Quintile portfolios on the primary factor
    qrets = quintile_returns(primary_signal, forward_rets, n_bins=5, min_assets=20)
    qsummary = quintile_summary(qrets)
    print("      quintile annual Sharpe (primary):")
    for q, v in qsummary.items():
        print(f"        {q}: mean={v['mean_ann']:+.3f} vol={v['vol_ann']:.3f} SR={v['sharpe']:+.3f}")

    # Benchmark factors
    mom12 = build_momentum_factor(returns, lookback=12)
    mom1 = build_mom1_reversal(returns)
    lowvol = build_lowvol_factor(returns, lookback=12)

    # Factor correlation — include BOTH sentiment variants
    factor_corr = factor_correlation_matrix({
        "sec_level": signal_level,
        "sec_change": signal_change,
        "mom12": mom12,
        "mom1_reversal": mom1,
        "lowvol": lowvol,
    })
    print("\n      factor correlation matrix:")
    print(factor_corr.round(3).to_string())

    # IC comparison across all factors
    all_factors = {
        "sec_level": signal_level,
        "sec_change": signal_change,
        "mom12": mom12,
        "mom1_reversal": mom1,
        "lowvol": lowvol,
    }
    ic_stats_by_factor = {}
    for name, panel in all_factors.items():
        ic_series = compute_ic_series(panel, forward_rets)
        ic_stats_by_factor[name] = compute_ic_stats(ic_series)
    ic_df = pd.DataFrame(ic_stats_by_factor).T
    print(f"\n      IC comparison:")
    print(ic_df.to_string())

    # ── 5. Plots ─────────────────────────────────────────
    print("[7/7] generating plots ...")
    _plot_ic_timeseries(ic, FIGS_DIR / "ic_timeseries.png")
    _plot_quintile_returns(qsummary, FIGS_DIR / "quintile_returns.png")
    _plot_factor_corr(factor_corr, FIGS_DIR / "factor_correlation.png")
    _plot_cumulative_ls(qrets, FIGS_DIR / "cumulative_ls.png")

    # ── 6. summary_stats.csv ─────────────────────────────
    summary = {
        "study": "task1_sec_nlp_sentiment",
        "primary_factor": primary_label,
        "n_tickers": int(manifest["ticker"].nunique()),
        "n_filings": int(len(manifest)),
        "n_months": int(len(ic)),
        "date_start": str(primary_signal.index.min().date()),
        "date_end": str(primary_signal.index.max().date()),
        **ic_stats,
        "q1_sharpe": qsummary.get("Q1", {}).get("sharpe", 0),
        "q5_sharpe": qsummary.get("Q5", {}).get("sharpe", 0),
        "ls_sharpe": qsummary.get("LS", {}).get("sharpe", 0),
        "ls_mean_ann": qsummary.get("LS", {}).get("mean_ann", 0),
        "corr_vs_mom12": factor_corr.loc[primary_row, "mom12"],
        "corr_vs_mom1": factor_corr.loc[primary_row, "mom1_reversal"],
        "corr_vs_lowvol": factor_corr.loc[primary_row, "lowvol"],
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(RESULTS_DIR / "summary_stats.csv", index=False)
    ic_df.to_csv(RESULTS_DIR / "ic_comparison.csv")
    factor_corr.to_csv(RESULTS_DIR / "factor_correlation.csv")

    print(f"\n[done] total elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"       results: {RESULTS_DIR}")
    return {"ic_stats": ic_stats, "ic_df": ic_df, "factor_corr": factor_corr,
            "qsummary": qsummary, "summary": summary}


# ────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────
def _plot_ic_timeseries(ic, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ic.plot(ax=ax, alpha=0.6, linewidth=1, label="IC (monthly)")
    ic.rolling(6).mean().plot(ax=ax, color="red", linewidth=2, label="6m rolling mean")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(ic.mean(), color="green", linestyle="--", linewidth=1,
               label=f"sample mean = {ic.mean():+.3f}")
    ax.set_title(f"SEC 10-K Net-Negativity Sentiment Factor — Spearman IC "
                 f"(n={len(ic)}, ICIR={(ic.mean()/ic.std()*np.sqrt(12)):.2f})")
    ax.set_ylabel("IC vs forward 1-month return")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_quintile_returns(qsummary, path):
    qs = [k for k in qsummary if k.startswith("Q")]
    qs.sort()
    means = [qsummary[k]["mean_ann"] for k in qs]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(qs, means, color=["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{m:+.2%}", ha="center", va="bottom" if m > 0 else "top", fontsize=9)
    ls = qsummary.get("LS", {}).get("mean_ann", 0)
    ax.set_title(f"Quintile Annualized Returns  |  L/S spread = {ls:+.2%}")
    ax.set_ylabel("Annualized mean return")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_factor_corr(corr, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:+.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    ax.set_title("Cross-Sectional Factor Correlation\n(avg Spearman across months)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_cumulative_ls(qrets, path):
    ls = qrets["LS"].dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    cum = (1 + ls).cumprod()
    ax.plot(cum.index, cum.values, color="navy", linewidth=1.5, label="L/S cumulative")
    ax.axhline(1.0, color="k", linewidth=0.5)
    total = cum.iloc[-1] - 1 if len(cum) else 0
    ax.set_title(f"Long-Short Quintile Spread — Cumulative Return  "
                 f"(total = {total:+.1%}, n={len(ls)} months)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
