"""13F holdings factor construction.

Inputs:
  long_panel  — output of build_holdings_panel:
                filer_name | filer_cik | accession | period_end |
                name | class | cusip | value | shrs | type
  cusip_map   — dict {cusip8 -> ticker} for our universe

Outputs:
  n_holders_panel  — DataFrame (period_end × ticker) of how many of
                     the tracked filers held the stock that quarter
  value_held_panel — DataFrame (period_end × ticker) of total dollar
                     value held by the tracked filers
  shrs_held_panel  — DataFrame (period_end × ticker) of total shares held

From these we derive:
  delta_n_holders  — quarter-over-quarter change in holder count
                     (the cleanest "smart money accumulation" signal)
  delta_value      — QoQ % change in dollar value held (mixes price moves
                     with position changes; less clean)
  delta_shares     — QoQ % change in shares held (clean position-change
                     signal, isolates flow from price)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def build_holdings_panels(
    long_panel: pd.DataFrame,
    cusip_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate the long-format holdings into three quarter × ticker panels.

    Steps:
      1. Map 13F's 9-digit CUSIPs to our 8-digit CRSP CUSIPs (drop check).
      2. Filter to holdings whose CUSIP is in our universe.
      3. Per (filer, period_end, ticker), aggregate sub-fund duplicates by
         summing shares and value.
      4. Per (period_end, ticker), aggregate across filers:
         - n_holders = count of distinct filers holding it
         - value_held = sum of dollar values
         - shrs_held  = sum of share counts
    """
    if long_panel.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    df = long_panel.copy()
    df["cusip8"] = df["cusip"].astype(str).str.upper().str[:8]
    df = df[df["cusip8"].isin(cusip_map.keys())].copy()
    df["ticker"] = df["cusip8"].map(cusip_map)

    # Aggregate sub-fund duplicates within a single filer's filing.
    per_filer = (
        df.groupby(["filer_name", "period_end", "ticker"])
          .agg(value=("value", "sum"), shrs=("shrs", "sum"))
          .reset_index()
    )

    # Cross-filer aggregation per (period_end, ticker).
    by_qtr = (
        per_filer.groupby(["period_end", "ticker"])
                 .agg(n_holders=("filer_name", "nunique"),
                      value_held=("value", "sum"),
                      shrs_held=("shrs", "sum"))
                 .reset_index()
    )

    n_panel = by_qtr.pivot(index="period_end", columns="ticker", values="n_holders").fillna(0).sort_index()
    v_panel = by_qtr.pivot(index="period_end", columns="ticker", values="value_held").fillna(0).sort_index()
    s_panel = by_qtr.pivot(index="period_end", columns="ticker", values="shrs_held").fillna(0).sort_index()
    return n_panel, v_panel, s_panel


def quarterly_signals(
    n_panel: pd.DataFrame,
    v_panel: pd.DataFrame,
    s_panel: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Build the candidate factor signals from the holdings panels.

    All signals are aligned to **quarter-end dates**. The signal at
    quarter-end T is computed using only filings whose period_end is
    T or earlier; downstream IC analysis applies forward returns over
    1m / 3m / 6m windows starting from T+1.

    delta_n_holders : N_t - N_{t-1}, raw count change
    delta_holders_pct : (N_t - N_{t-1}) / max(N_{t-1}, 1)
    delta_shares_pct  : (S_t - S_{t-1}) / max(S_{t-1}, 1) — capped at +100%
                        to limit single-stock outliers
    delta_value_pct   : (V_t - V_{t-1}) / max(V_{t-1}, 1) — for completeness
                        but contaminated by price moves
    """
    out = {}
    out["delta_n_holders"] = n_panel.diff()
    out["delta_holders_pct"] = n_panel.diff() / n_panel.shift(1).clip(lower=1)

    # Cap pct signals at ±1 to keep IC computation robust to outliers
    out["delta_shares_pct"] = (s_panel.diff() / s_panel.shift(1).clip(lower=1)).clip(-1, 1)
    out["delta_value_pct"]  = (v_panel.diff() / v_panel.shift(1).clip(lower=1)).clip(-1, 1)
    return out


def expand_quarterly_to_monthly(
    quarterly_signal: pd.DataFrame,
    monthly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill a quarter-end signal to a monthly index.

    13F is filed at quarter-end (with a 45-day reporting lag). For
    cross-sectional IC computation against monthly forward returns,
    we hold each quarter's signal value constant for the next 3 months.
    """
    # Use a Timestamp index for join compatibility
    sig = quarterly_signal.sort_index()
    return sig.reindex(monthly_index, method="ffill")
