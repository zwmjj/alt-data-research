#!/usr/bin/env python
"""Task 4 end-to-end runner — combined alternative-data portfolio.

Compares:
  - Each individual alt-data signal (Task 1, 2, 3)
  - Combined alt-data factor (equal-weight + IC-weighted)
  - Traditional factor stack (mom12, mom1_reversal, lowvol)
  - Combined alt-data ⊕ traditional (full kitchen sink)

Outputs:
  - IC × forward horizon for every signal
  - Quintile portfolio Sharpe table
  - Cumulative L/S overlay (one chart, all factors)
  - Drawdown chart of the combined factor
  - Factor correlation heat map (8x8)
  - Final summary_stats.csv binding the project together
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
from src.factor_combined import build_combined_factor
from src.factor_analysis import (
    compute_ic_series, compute_ic_stats,
    quintile_returns, quintile_summary,
    build_momentum_factor, build_lowvol_factor, build_mom1_reversal,
    factor_correlation_matrix,
)

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()

    # ── 1. Universe + returns ────────────────────────────
    print("[1/6] loading universe + returns ...")
    sp100 = get_sp100_tickers()
    returns = fetch_returns(sp100["ticker"].tolist(), start="2017-12-01", end="2025-06-30")
    forward_rets = returns.shift(-1)
    print(f"      returns: {returns.shape}")

    # ── 2. Build combined factor (both weightings) ──────
    print("\n[2/6] building combined factor (equal + IC weighting) ...")
    eq = build_combined_factor(forward_rets, weighting="equal")
    ic = build_combined_factor(forward_rets, weighting="ic")

    # ── 3. Define the signal universe to compare ────────
    mom12 = build_momentum_factor(returns, lookback=12)
    mom1 = build_mom1_reversal(returns)
    lowvol = build_lowvol_factor(returns, lookback=12)

    signals = {
        "task1_10k_signed":     eq["task1"],
        "task2_13f_signed":     eq["task2"],
        "task3_8k_signed":      eq["task3"],
        "combined_equal":       eq["combined"],
        "combined_ic_weighted": ic["combined"],
        "mom12":                mom12,
        "mom1_reversal":        mom1,
        "lowvol":               lowvol,
    }

    # ── 4. IC, quintile, drawdown for each signal ───────
    print("\n[3/6] computing IC + quintile + drawdown for each signal ...")
    summary_rows = []
    qrets_by_signal = {}
    print(f"\n      {'signal':<22} {'IC_1m':>9} {'ICIR':>8} {'t':>8} {'hit':>8} {'Q5_SR':>8} {'LS_SR':>8}")
    for name, sig in signals.items():
        ic_series = compute_ic_series(sig, forward_rets)
        ic_stats = compute_ic_stats(ic_series)
        qrets = quintile_returns(sig, forward_rets, n_bins=5, min_assets=20)
        qrets_by_signal[name] = qrets
        qsum = quintile_summary(qrets)
        ls_sr = qsum.get("LS", {}).get("sharpe", 0)
        ls_mean = qsum.get("LS", {}).get("mean_ann", 0)
        q5_sr = qsum.get("Q5", {}).get("sharpe", 0)
        # Cumulative L/S drawdown
        ls = qrets["LS"].dropna()
        if len(ls) > 1:
            cum = (1 + ls).cumprod()
            dd = (cum / cum.cummax() - 1).min()
        else:
            dd = 0.0

        summary_rows.append({
            "signal": name,
            "ic_mean": ic_stats["ic_mean"],
            "icir": ic_stats["icir"],
            "t_stat": ic_stats["t_stat"],
            "hit_rate": ic_stats["hit_rate"],
            "ls_sharpe": ls_sr,
            "ls_mean_ann": ls_mean,
            "max_drawdown": float(dd),
            "n_obs": ic_stats["n_obs"],
        })
        print(f"      {name:<22} {ic_stats['ic_mean']:>+9.4f} "
              f"{ic_stats['icir']:>+8.3f} {ic_stats['t_stat']:>+8.2f} "
              f"{ic_stats['hit_rate']:>8.1%} {q5_sr:>+8.3f} {ls_sr:>+8.3f}")

    summary_df = pd.DataFrame(summary_rows)

    # ── 5. Factor correlation matrix ────────────────────
    print("\n[4/6] computing factor correlation matrix ...")
    factor_corr = factor_correlation_matrix(signals)
    print(factor_corr.round(3).to_string())

    # ── 6. Plots ─────────────────────────────────────────
    print("\n[5/6] generating plots ...")
    _plot_cumulative_ls_overlay(qrets_by_signal, FIGS / "task4_cumulative_overlay.png")
    _plot_ic_comparison(summary_df, FIGS / "task4_ic_comparison.png")
    _plot_drawdown(qrets_by_signal["combined_equal"]["LS"].dropna(),
                   FIGS / "task4_drawdown_combined.png")
    _plot_factor_corr(factor_corr, FIGS / "task4_factor_correlation.png")
    _plot_rolling_sharpe(qrets_by_signal, FIGS / "task4_rolling_sharpe.png")

    # ── 7. Final summary CSV ────────────────────────────
    print("\n[6/6] writing summary_stats.csv ...")
    summary_df.to_csv(RESULTS / "summary_stats_task4.csv", index=False)
    factor_corr.to_csv(RESULTS / "factor_correlation_task4.csv")

    # Final master summary tying all 4 tasks
    final = {
        "task1_10k_sharpe":         summary_df.set_index("signal").loc["task1_10k_signed", "ls_sharpe"],
        "task2_13f_sharpe":         summary_df.set_index("signal").loc["task2_13f_signed", "ls_sharpe"],
        "task3_8k_sharpe":          summary_df.set_index("signal").loc["task3_8k_signed", "ls_sharpe"],
        "combined_equal_sharpe":    summary_df.set_index("signal").loc["combined_equal", "ls_sharpe"],
        "combined_ic_sharpe":       summary_df.set_index("signal").loc["combined_ic_weighted", "ls_sharpe"],
        "mom12_sharpe":             summary_df.set_index("signal").loc["mom12", "ls_sharpe"],
        "lowvol_sharpe":            summary_df.set_index("signal").loc["lowvol", "ls_sharpe"],
        "combined_ic_mean":         summary_df.set_index("signal").loc["combined_equal", "ic_mean"],
        "combined_icir":            summary_df.set_index("signal").loc["combined_equal", "icir"],
        "combined_t_stat":          summary_df.set_index("signal").loc["combined_equal", "t_stat"],
        "combined_max_dd":          summary_df.set_index("signal").loc["combined_equal", "max_drawdown"],
    }
    pd.DataFrame([final]).to_csv(RESULTS / "summary_master.csv", index=False)

    print(f"\n[done] elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"       results: {RESULTS}")
    return {"summary": summary_df, "factor_corr": factor_corr, "final": final}


# ────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────
def _plot_cumulative_ls_overlay(qrets_by_signal, path):
    """One chart, all factors' L/S cumulative growth overlaid."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = {
        "task1_10k_signed":     ("#fdae61", 1.2, "--"),
        "task2_13f_signed":     ("#fdae61", 1.2, "--"),
        "task3_8k_signed":      ("#fdae61", 1.2, "--"),
        "combined_equal":       ("#1a9641", 2.5, "-"),
        "combined_ic_weighted": ("#006837", 2.5, "-"),
        "mom12":                ("#2c7fb8", 1.0, ":"),
        "mom1_reversal":        ("#2c7fb8", 1.0, ":"),
        "lowvol":               ("#2c7fb8", 1.0, ":"),
    }
    for name, qrets in qrets_by_signal.items():
        ls = qrets["LS"].dropna()
        if len(ls) < 2:
            continue
        cum = (1 + ls).cumprod()
        color, lw, ls_style = palette.get(name, ("gray", 1.0, "-"))
        ax.plot(cum.index, cum.values, color=color, linewidth=lw,
                linestyle=ls_style, label=f"{name} ({(cum.iloc[-1]-1)*100:+.0f}%)")
    ax.axhline(1.0, color="k", linewidth=0.5)
    ax.set_title("Long-Short Quintile Cumulative Returns — All Factors Overlay")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_ic_comparison(summary_df, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    df = summary_df.copy().set_index("signal")
    order = ["task1_10k_signed", "task2_13f_signed", "task3_8k_signed",
             "combined_equal", "combined_ic_weighted",
             "mom12", "mom1_reversal", "lowvol"]
    df = df.loc[[s for s in order if s in df.index]]

    colors = ["#fdae61"] * 3 + ["#1a9641", "#006837"] + ["#2c7fb8"] * 3
    ax1.bar(range(len(df)), df["icir"].values, color=colors[:len(df)])
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df.index, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("ICIR (annualized)")
    ax1.set_title("Information Ratio of IC")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(range(len(df)), df["ls_sharpe"].values, color=colors[:len(df)])
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df.index, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("L/S Sharpe (annualized)")
    ax2.set_title("Long-Short Quintile Sharpe")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_drawdown(ls_series, path):
    cum = (1 + ls_series).cumprod()
    dd = cum / cum.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(dd.index, 0, dd.values, color="darkred", alpha=0.5)
    ax.plot(dd.index, dd.values, color="darkred", linewidth=1)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title(f"Combined alt-data L/S Drawdown  (max DD = {dd.min():.1%})")
    ax.set_ylabel("Drawdown from peak")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_factor_corr(corr, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    ax.set_title("Cross-Sectional Factor Correlation\nIndividual alt-data + combined + traditional")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_rolling_sharpe(qrets_by_signal, path):
    """24-month rolling Sharpe of the combined factor + key benchmarks."""
    keys = ["combined_equal", "combined_ic_weighted", "mom12", "lowvol"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k in keys:
        if k not in qrets_by_signal:
            continue
        ls = qrets_by_signal[k]["LS"].dropna()
        if len(ls) < 24:
            continue
        rolling = ls.rolling(24).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0,
            raw=False,
        )
        ax.plot(rolling.index, rolling.values, label=k, linewidth=1.5,
                alpha=0.9 if "combined" in k else 0.6)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("24-Month Rolling Sharpe of L/S Quintile Spread")
    ax.set_ylabel("Rolling annualized Sharpe")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
