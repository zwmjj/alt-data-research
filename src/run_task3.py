#!/usr/bin/env python
"""Task 3 end-to-end runner — 8-K news sentiment factor.

Pipeline:
  1. Walk DATA_RAW_8K to build a manifest of all downloaded 8-Ks
  2. Parse each filing → body + EX-99 + items + dates
  3. LM-score every filing (full corpus)
  4. FinBERT-score a 300-filing subsample (spec compliance + comparison)
  5. Build monthly signal panels (sentiment level, sentiment Δ, count, earnings-only count)
  6. IC at 1m / 3m forward returns; quintile sort
  7. Factor correlation vs Task 1 (sentiment), Task 2 (13F), traditional
  8. Plots + summary CSV
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
from src.data_fetcher_8k import list_8k_paths, parse_all_filings
from src.factor_8k_news import (
    score_filings_lm, score_filings_finbert,
    build_monthly_panel, build_count_panel,
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


def main(skip_finbert: bool = False):
    t0 = time.time()

    # ── 1. Universe + returns ────────────────────────────
    print("[1/8] loading universe + returns ...")
    sp100 = get_sp100_tickers()
    returns = fetch_returns(sp100["ticker"].tolist(), start="2017-12-01", end="2025-06-30")
    print(f"      returns: {returns.shape}")

    # ── 2. Manifest of 8-K filings ───────────────────────
    print("[2/8] scanning 8-K download tree ...")
    manifest = list_8k_paths()
    print(f"      {len(manifest)} filings across {manifest['ticker'].nunique()} tickers")

    # ── 3. Parse + LM score ──────────────────────────────
    cached_scored = PROCESSED / "filings_8k_lm.parquet"
    if cached_scored.exists():
        print(f"[3/8] loading cached LM-scored filings: {cached_scored}")
        lm_scored = pd.read_parquet(cached_scored)
    else:
        print("[3/8] parsing all 8-K filings ...")
        parsed = parse_all_filings(manifest)
        print(f"      parsed {len(parsed)} filings")

        print("[3b/8] LM scoring (full corpus) ...")
        lm_scored = score_filings_lm(parsed)
        # Drop the heavy text columns before persisting — we already extracted everything
        lm_scored_lite = lm_scored.drop(columns=["body", "ex99"], errors="ignore")
        lm_scored_lite.to_parquet(cached_scored)
        lm_scored = lm_scored_lite
    print(f"      LM-scored shape: {lm_scored.shape}")

    # ── 4. FinBERT on a subset (optional, for spec compliance) ──
    cached_finbert = PROCESSED / "filings_8k_finbert.parquet"
    if not skip_finbert and not cached_finbert.exists():
        print("[4/8] running FinBERT on 300-filing subsample ...")
        # Need body+ex99 — re-parse the sampled rows
        from src.data_fetcher_8k import parse_8k_filing
        sample_meta = manifest.sample(min(300, len(manifest)), random_state=42).reset_index(drop=True)
        sample_full = sample_meta.copy()
        bodies, ex99s = [], []
        for _, row in sample_meta.iterrows():
            m = parse_8k_filing(Path(row["path"]))
            bodies.append(m["body"]); ex99s.append(m["ex99"])
        sample_full["body"] = bodies
        sample_full["ex99"] = ex99s
        # Bring in filed_date / items via the LM panel
        sample_full = sample_full.merge(
            lm_scored[["accession", "ticker", "filed_date", "items"]],
            on=["accession", "ticker"], how="left",
        )
        finbert_scored = score_filings_finbert(sample_full, sample_size=300)
        finbert_scored.drop(columns=["body", "ex99"], errors="ignore").to_parquet(cached_finbert)
        print(f"      FinBERT scored {len(finbert_scored)} filings, mean polarity = "
              f"{finbert_scored['finbert_polarity'].mean():+.4f}")
    elif cached_finbert.exists():
        print(f"[4/8] loading cached FinBERT scores")
    else:
        print("[4/8] skipping FinBERT (skip_finbert=True)")

    # ── 5. Build signal panels ───────────────────────────
    print("[5/8] building monthly panels ...")
    sentiment_level = -build_monthly_panel(lm_scored, metric="net_neg")  # flip: high = bullish
    sentiment_change = sentiment_level.diff(1)  # MoM change in sentiment
    count_all = build_count_panel(lm_scored)
    count_earnings = build_count_panel(lm_scored, item_filter=["2.02"])
    print(f"      sentiment_level panel: {sentiment_level.shape}, non-null cells: {sentiment_level.notna().sum().sum()}")
    print(f"      count_all: total filings in panel = {int(count_all.sum().sum())}")
    print(f"      count_earnings: total = {int(count_earnings.sum().sum())}")

    # ── 6. IC analysis at 1m / 3m forward ───────────────
    print("[6/8] computing IC at forward 1m/3m ...")
    forward_1m = returns.shift(-1)
    forward_3m = ((1 + returns).rolling(3).apply(lambda x: np.prod(x), raw=True) - 1).shift(-3)

    signals = {
        "sentiment_level": sentiment_level,
        "sentiment_change": sentiment_change,
        "count_all": count_all,
        "count_earnings": count_earnings,
    }

    horizon_ic = {}
    print(f"\n      {'signal':<20} {'horizon':<6} {'IC':>8} {'ICIR':>8} {'t':>8} {'hit':>8} {'n':>5}")
    for sname, sig in signals.items():
        horizon_ic[sname] = {}
        for hname, fwd in [("1m", forward_1m), ("3m", forward_3m)]:
            ic = compute_ic_series(sig, fwd)
            stats = compute_ic_stats(ic)
            horizon_ic[sname][hname] = stats
            print(f"      {sname:<20} {hname:<6} {stats['ic_mean']:>+8.4f} "
                  f"{stats['icir']:>+8.3f} {stats['t_stat']:>+8.2f} "
                  f"{stats['hit_rate']:>8.1%} {stats['n_obs']:>5d}")

    # ── 7. Pick best signal × horizon, quintile + corr ─
    print("\n[7/8] selecting strongest signal by |ICIR| ...")
    # Restrict the quintile-analysis "winner" to *continuous* signals.
    # `count_*` panels are sparse step functions (most cells are 0
    # because most stocks have no 8-K in any given month), so a
    # quintile sort over them produces degenerate bins where Q3..Q5
    # collapse to a single value. The continuous sentiment signals
    # have well-defined quintiles. Both `count_*` and `sentiment_*`
    # IC numbers are still reported in the matrix above.
    continuous_signals = {"sentiment_level", "sentiment_change"}
    best = max(
        ((s, h, horizon_ic[s][h])
         for s in horizon_ic if s in continuous_signals
         for h in horizon_ic[s]),
        key=lambda x: abs(x[2]["icir"]),
    )
    bs, bh, bstats = best
    print(f"      WINNER (continuous): {bs} @ {bh}  ICIR={bstats['icir']:+.3f}  t={bstats['t_stat']:+.2f}")

    best_signal = signals[bs]
    best_fwd = forward_1m if bh == "1m" else forward_3m
    qrets = quintile_returns(best_signal, best_fwd, n_bins=5, min_assets=20)
    qsummary = quintile_summary(qrets)
    print(f"      quintile annual Sharpe at {bh}:")
    for q, v in qsummary.items():
        print(f"        {q}: mean={v['mean_ann']:+.3f} vol={v['vol_ann']:.3f} SR={v['sharpe']:+.3f}")

    # Factor correlation vs Task 1 + Task 2 + traditional
    mom12 = build_momentum_factor(returns, lookback=12)
    mom1 = build_mom1_reversal(returns)
    lowvol = build_lowvol_factor(returns, lookback=12)

    corr_inputs = {
        "8k_best": best_signal,
        "8k_count_earnings": count_earnings,
        "mom12": mom12,
        "mom1_reversal": mom1,
        "lowvol": lowvol,
    }
    # Pull Task 1 sentiment if available
    sf_path = PROCESSED / "filings_scored.parquet"
    if sf_path.exists():
        from src.factor_analysis import build_signal_panel
        scored_10k = pd.read_parquet(sf_path)
        task1_sig = -build_signal_panel(scored_10k, "net_neg", "2018-01-31", "2025-06-30", 12)
        corr_inputs["task1_sec_sentiment"] = task1_sig
    # Pull Task 2 best 13F if available
    h13f_path = PROCESSED / "holdings_13f_long.parquet"
    if h13f_path.exists():
        from src.data_fetcher_13f import get_filer_ciks, get_cusip_ticker_map
        from src.factor_13f import build_holdings_panels, quarterly_signals, expand_quarterly_to_monthly
        long_panel = pd.read_parquet(h13f_path)
        cusip_map = get_cusip_ticker_map(sp100["ticker"].tolist())
        n_panel, v_panel, s_panel = build_holdings_panels(long_panel, cusip_map)
        sigs_13f = quarterly_signals(n_panel, v_panel, s_panel)
        task2_sig = expand_quarterly_to_monthly(sigs_13f["delta_shares_pct"], best_signal.index)
        corr_inputs["task2_13f_delta_shrs"] = task2_sig

    factor_corr = factor_correlation_matrix(corr_inputs)
    print("\n      factor correlation matrix:")
    print(factor_corr.round(3).to_string())

    # ── 8. Plots + CSV ───────────────────────────────────
    print("\n[8/8] generating plots ...")
    _plot_horizon_ic(horizon_ic, FIGS / "8k_horizon_ic.png")
    _plot_quintile_returns(qsummary, FIGS / "8k_quintile_returns.png", title=f"8-K {bs} @ {bh}")
    _plot_factor_corr(factor_corr, FIGS / "8k_factor_correlation.png")
    _plot_cumulative_ls(qrets, FIGS / "8k_cumulative_ls.png", title=f"8-K L/S — {bs} @ {bh}")
    _plot_filings_per_month(count_all, FIGS / "8k_filings_per_month.png")

    summary = {
        "study": "task3_8k_news_sentiment",
        "primary_signal": bs,
        "primary_horizon": bh,
        "n_tickers": int(manifest["ticker"].nunique()),
        "n_filings": int(len(manifest)),
        **bstats,
        "q1_sharpe": qsummary.get("Q1", {}).get("sharpe", 0),
        "q5_sharpe": qsummary.get("Q5", {}).get("sharpe", 0),
        "ls_sharpe": qsummary.get("LS", {}).get("sharpe", 0),
        "ls_mean_ann": qsummary.get("LS", {}).get("mean_ann", 0),
        "corr_vs_mom12": factor_corr.loc["8k_best", "mom12"],
        "corr_vs_lowvol": factor_corr.loc["8k_best", "lowvol"],
    }
    if "task1_sec_sentiment" in factor_corr.columns:
        summary["corr_vs_task1"] = factor_corr.loc["8k_best", "task1_sec_sentiment"]
    if "task2_13f_delta_shrs" in factor_corr.columns:
        summary["corr_vs_task2"] = factor_corr.loc["8k_best", "task2_13f_delta_shrs"]
    pd.DataFrame([summary]).to_csv(RESULTS / "summary_stats_task3.csv", index=False)
    factor_corr.to_csv(RESULTS / "factor_correlation_task3.csv")

    # IC matrix CSV
    ic_rows = []
    for sname, hs in horizon_ic.items():
        row = {"signal": sname}
        for h, st in hs.items():
            row[f"ic_{h}"] = st["ic_mean"]
            row[f"icir_{h}"] = st["icir"]
            row[f"t_{h}"] = st["t_stat"]
        ic_rows.append(row)
    pd.DataFrame(ic_rows).to_csv(RESULTS / "ic_matrix_task3.csv", index=False)

    print(f"\n[done] elapsed: {(time.time()-t0)/60:.1f} min")
    return {"horizon_ic": horizon_ic, "best": best, "qsummary": qsummary,
            "factor_corr": factor_corr, "summary": summary}


# ────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────
def _plot_horizon_ic(horizon_ic, path):
    sig_names = list(horizon_ic.keys())
    horizons = ["1m", "3m"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    x = np.arange(len(sig_names))
    colors = ["#1a9641", "#d7191c"]
    for i, h in enumerate(horizons):
        icirs = [horizon_ic[s][h]["icir"] for s in sig_names]
        ax.bar(x + i*width, icirs, width, label=f"forward {h}", color=colors[i], alpha=0.85)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(sig_names, rotation=15, ha="right")
    ax.set_ylabel("ICIR (annualized)")
    ax.set_title("8-K News Factors — IC Information Ratio by Forward Horizon")
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
    n = len(corr)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i,j]:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    ax.set_title("Cross-Sectional Factor Correlation\n8-K + Tasks 1-2 + traditional")
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
    ax.plot(cum.index, cum.values, color="purple", linewidth=1.5, label="L/S cumulative")
    ax.axhline(1.0, color="k", linewidth=0.5)
    total = cum.iloc[-1] - 1
    ax.set_title(f"{title}  ({total:+.1%} total, n={len(ls)} months)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_filings_per_month(count_all, path):
    monthly_total = count_all.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly_total.index, monthly_total.values, color="steelblue", alpha=0.8, width=20)
    ax.set_title(f"Total S&P 100 8-K filings per month  (avg = {monthly_total.mean():.0f})")
    ax.set_ylabel("# of 8-K filings")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-finbert", action="store_true",
                        help="Skip the FinBERT subset scoring (faster).")
    args = parser.parse_args()
    main(skip_finbert=args.skip_finbert)
