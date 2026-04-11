"""Combine the three alternative-data signals into a single alpha block.

Inputs: cached parquet panels written by run_task1 / run_task2 / run_task3.
Outputs: a monthly (date × ticker) panel that downstream factor_analysis
treats just like any other signal.

Combination protocol:
  1. **Sign-flip to predictive direction.** Each individual signal's
     directional sign is determined by computing its in-sample IC vs
     forward returns and multiplying by `sign(IC)`. Without this, blending
     a +IC signal and a -IC signal partially cancels out.
  2. **Cross-sectional z-score per period.** Each signal is z-scored
     across the cross-section at each rebalance date so different units
     (sentiment ratio vs holding count vs share % delta) are commensurable.
  3. **Blend.**
       - `equal_weight` averages the z-scored signals 1/N
       - `ic_weighted` uses |ICIR| weights from the in-sample period to
         tilt toward stronger components
  4. **Forward-fill alignment.** Components with different cadences
     (10-K = annual, 13F = quarterly, 8-K = monthly) are reindexed to
     a common monthly index. Cells where any component is missing are
     dropped from the corresponding cross-section to keep the blend
     unbiased.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


# ────────────────────────────────────────────────────────────
# 1. Load each task's primary signal as a (date × ticker) panel
# ────────────────────────────────────────────────────────────
def load_task1_signal(start: str = "2018-01-31", end: str = "2025-06-30") -> pd.DataFrame:
    """Task 1 — 10-K LM sentiment level, forward-filled 12 months."""
    from src.factor_analysis import build_signal_panel
    scored = pd.read_parquet(PROCESSED / "filings_scored.parquet")
    # Note: Task 1 already builds with `-net_neg` to make higher = bullish.
    # We DON'T pre-flip here — let the combiner sign-flip based on IC.
    return build_signal_panel(scored, "net_neg", start=start, end=end, hold_months=12)


def load_task2_signal(start: str = "2018-01-31", end: str = "2025-06-30") -> pd.DataFrame:
    """Task 2 — 13F delta_shares_pct, expanded from quarterly to monthly."""
    from src.data_fetcher import get_sp100_tickers
    from src.data_fetcher_13f import get_filer_ciks, get_cusip_ticker_map
    from src.factor_13f import build_holdings_panels, quarterly_signals, expand_quarterly_to_monthly

    long_panel = pd.read_parquet(PROCESSED / "holdings_13f_long.parquet")
    sp100 = get_sp100_tickers()
    cusip_map = get_cusip_ticker_map(sp100["ticker"].tolist())
    n_panel, v_panel, s_panel = build_holdings_panels(long_panel, cusip_map)
    sigs = quarterly_signals(n_panel, v_panel, s_panel)
    monthly_idx = pd.date_range(start, end, freq="ME")
    return expand_quarterly_to_monthly(sigs["delta_shares_pct"], monthly_idx)


def load_task3_signal(start: str = "2018-01-31", end: str = "2025-06-30") -> pd.DataFrame:
    """Task 3 — 8-K sentiment_level monthly panel."""
    from src.factor_8k_news import build_monthly_panel
    scored = pd.read_parquet(PROCESSED / "filings_8k_lm.parquet")
    return build_monthly_panel(scored, metric="net_neg", start=start, end=end)


# ────────────────────────────────────────────────────────────
# 2. Sign + standardize + blend
# ────────────────────────────────────────────────────────────
def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (date) across the cross-section of tickers.

    NaN values stay NaN — we don't impute. Rows with std == 0
    (all-equal cross-section) collapse to all-zero, which is the
    correct neutral signal for that period.
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0, np.nan)
    z = panel.sub(mean, axis=0).div(std, axis=0)
    return z.fillna(0).where(panel.notna())  # restore NaN mask


def signed_zscore_signal(
    panel: pd.DataFrame,
    forward_rets: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    """Z-score the panel cross-sectionally, then multiply by sign(IC)
    so the resulting signal is in its 'predictive' direction.

    Returns (signed_z_panel, ic_mean).
    """
    from src.factor_analysis import compute_ic_series
    z = cross_sectional_zscore(panel)
    ic_series = compute_ic_series(z, forward_rets)
    ic_mean = float(ic_series.mean()) if len(ic_series) else 0.0
    direction = float(np.sign(ic_mean)) if abs(ic_mean) > 1e-6 else 1.0
    return z * direction, ic_mean


def blend_signals(
    components: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Average a dict of {name: signed_z_panel}.

    `weights` defaults to equal weights. Components are aligned on
    intersection of dates and tickers; missing cells reduce that
    period's effective weight (we re-normalize across the available
    components per cell so the blend is unbiased when one source is
    missing — e.g. months where 8-K had no filing for a stock).
    """
    if not components:
        return pd.DataFrame()

    # Align on intersection of dates and tickers
    dates = None
    cols = None
    for panel in components.values():
        dates = panel.index if dates is None else dates.intersection(panel.index)
        cols = panel.columns if cols is None else cols.intersection(panel.columns)

    aligned = {n: p.loc[dates, cols] for n, p in components.items()}
    w = pd.Series(weights or {n: 1.0 / len(aligned) for n in aligned})
    w = w.reindex(list(aligned.keys())).fillna(0)
    w = w / w.sum() if w.sum() > 0 else w

    # Per-cell weighted average over the components present at that cell.
    stacked = pd.concat(aligned.values(), axis=1, keys=aligned.keys())
    # `stacked` is a column-MultiIndex (component, ticker)
    w_arr = w.values  # length = number of components

    out = pd.DataFrame(0.0, index=dates, columns=cols)
    weight_sum = pd.DataFrame(0.0, index=dates, columns=cols)
    for i, (name, panel) in enumerate(aligned.items()):
        contrib = panel.fillna(0) * w_arr[i]
        mask = panel.notna().astype(float) * w_arr[i]
        out = out + contrib
        weight_sum = weight_sum + mask
    # Re-normalize so cells with fewer components are not down-weighted
    out = out.div(weight_sum.replace(0, np.nan))
    return out


def build_combined_factor(
    forward_rets: pd.DataFrame,
    weighting: str = "equal",
    start: str = "2018-01-31",
    end: str = "2025-06-30",
) -> Dict:
    """End-to-end builder.

    Returns a dict with:
      task1, task2, task3   — individual signed z-score panels
      combined              — blended panel (equal or IC-weighted)
      ic_means              — per-component IC mean (used for sign + weight)
      weights               — final blend weights
    """
    print("[combined] loading task signals ...")
    p1 = load_task1_signal(start, end)
    p2 = load_task2_signal(start, end)
    p3 = load_task3_signal(start, end)
    print(f"  task1 (10-K) panel: {p1.shape}, non-null = {p1.notna().sum().sum()}")
    print(f"  task2 (13F) panel: {p2.shape}, non-null = {p2.notna().sum().sum()}")
    print(f"  task3 (8-K) panel: {p3.shape}, non-null = {p3.notna().sum().sum()}")

    s1, ic1 = signed_zscore_signal(p1, forward_rets)
    s2, ic2 = signed_zscore_signal(p2, forward_rets)
    s3, ic3 = signed_zscore_signal(p3, forward_rets)
    print(f"[combined] signed IC means: t1={ic1:+.4f}  t2={ic2:+.4f}  t3={ic3:+.4f}")

    components = {"task1_10k": s1, "task2_13f": s2, "task3_8k": s3}

    if weighting == "equal":
        weights = {n: 1 / len(components) for n in components}
    elif weighting == "ic":
        # |IC| weights, normalized
        abs_ics = {"task1_10k": abs(ic1), "task2_13f": abs(ic2), "task3_8k": abs(ic3)}
        s = sum(abs_ics.values())
        weights = {n: v / s for n, v in abs_ics.items()} if s > 0 \
                  else {n: 1 / 3 for n in components}
    else:
        raise ValueError(f"unknown weighting '{weighting}'")
    print(f"[combined] weights ({weighting}): {weights}")

    combined = blend_signals(components, weights)
    return {
        "task1": s1, "task2": s2, "task3": s3,
        "combined": combined,
        "ic_means": {"task1_10k": ic1, "task2_13f": ic2, "task3_8k": ic3},
        "weights": weights,
    }
