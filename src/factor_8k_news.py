"""8-K news sentiment factor.

For each S&P 100 stock × month:
  - count of 8-K filings filed that month (news flow intensity)
  - average LM net_neg sentiment of those filings (combined body + EX-99)
  - count of earnings filings (Item 2.02) specifically
  - month-over-month change in average sentiment

The LM dictionary is the workhorse — fast, free, well-established for
SEC text analysis. We also expose a `score_with_finbert` function that
applies FinBERT to a subset of filings (the spec mentions FinBERT, so
we honor that on a sampled cohort even though LM scales better for
the full corpus).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.nlp_signals import get_lm, lm_score


# ────────────────────────────────────────────────────────────
# 1. LM scoring of parsed filings
# ────────────────────────────────────────────────────────────
def score_filings_lm(parsed: pd.DataFrame) -> pd.DataFrame:
    """Apply LM scoring to body + EX-99 of every parsed 8-K.

    Returns the input DataFrame plus columns:
      n_tokens, pos_count, neg_count, pos_ratio, neg_ratio, net_neg,
      polarity, subjectivity
    """
    rows = []
    for i, row in parsed.iterrows():
        text = (row.get("body") or "") + " " + (row.get("ex99") or "")
        scores = lm_score(text)
        rows.append({**row.to_dict(), **scores})
        if (i + 1) % 500 == 0:
            print(f"  LM-scored {i+1}/{len(parsed)} filings")
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# 2. Optional: FinBERT scoring on a subset
# ────────────────────────────────────────────────────────────
def score_filings_finbert(
    parsed: pd.DataFrame,
    sample_size: int = 300,
    max_tokens: int = 510,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Apply FinBERT to a random subset of filings.

    FinBERT is a BERT model fine-tuned on financial text (Araci 2019).
    Returns probabilities for {positive, negative, neutral}. We use
    `ProsusAI/finbert` which is the canonical version.

    We restrict to a random subsample because FinBERT inference is
    ~5-10× slower than LM dictionary even on GPU, and the spec only
    asks for FinBERT as a *comparison* point, not a full-corpus pass.
    """
    if len(parsed) == 0:
        return pd.DataFrame()

    sample = parsed.sample(min(sample_size, len(parsed)), random_state=42).reset_index(drop=True)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading FinBERT (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()

    # FinBERT label order from config: ['positive', 'negative', 'neutral']
    labels = ["positive", "negative", "neutral"]

    pos_probs, neg_probs, neu_probs = [], [], []
    for start in range(0, len(sample), batch_size):
        chunk = sample.iloc[start:start + batch_size]
        texts = []
        for _, row in chunk.iterrows():
            t = (row.get("body") or "") + " " + (row.get("ex99") or "")
            # FinBERT cap is 512 tokens. We use the *first* max_tokens tokens
            # of the press release; for earnings releases this is the
            # headline + executive summary, which is where sentiment lives.
            texts.append(t[:max_tokens * 6] if t else "")  # ~6 chars/token avg

        if not any(texts):
            for _ in range(len(chunk)):
                pos_probs.append(np.nan); neg_probs.append(np.nan); neu_probs.append(np.nan)
            continue

        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True,
                               max_length=max_tokens, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            pos_probs.append(float(p[0]))
            neg_probs.append(float(p[1]))
            neu_probs.append(float(p[2]))

        if (start // batch_size + 1) % 10 == 0:
            print(f"    FinBERT batch {start + len(chunk)}/{len(sample)}")

    sample["finbert_pos"] = pos_probs
    sample["finbert_neg"] = neg_probs
    sample["finbert_neu"] = neu_probs
    sample["finbert_polarity"] = sample["finbert_pos"] - sample["finbert_neg"]
    return sample


# ────────────────────────────────────────────────────────────
# 3. Monthly aggregation per ticker
# ────────────────────────────────────────────────────────────
def build_monthly_panel(
    scored: pd.DataFrame,
    metric: str = "net_neg",
    start: str = "2018-01-31",
    end: str = "2025-06-30",
) -> pd.DataFrame:
    """Aggregate per-filing scores into a (month-end × ticker) panel.

    For each (ticker, month) cell, we take the **average** of the
    metric across all 8-Ks filed in that month. Months with no filings
    are NaN — we deliberately do NOT forward-fill, because the absence
    of news is itself information (vs Task 1, where 10-Ks are mandatory
    annual events that we forward-filled to bridge gaps).
    """
    df = scored.copy()
    df = df.dropna(subset=["filed_date"])
    # `MonthEnd(0)` preserves the time component of filed_date (e.g.
    # 16:30:17), so each filing ends up with a unique full timestamp
    # instead of a clean month-end DATE — which means the reindex to
    # our month-end DatetimeIndex matches NOTHING and the panel comes
    # out all-NaN. Normalize to midnight first so all filings in the
    # same month collapse to the same key.
    df["month_end"] = df["filed_date"].dt.normalize() + pd.offsets.MonthEnd(0)
    grouped = (df.groupby(["month_end", "ticker"])[metric]
                 .mean()
                 .unstack("ticker"))
    full_idx = pd.date_range(start, end, freq="ME")
    grouped = grouped.reindex(full_idx)
    return grouped


def build_count_panel(
    scored: pd.DataFrame,
    item_filter: Optional[List[str]] = None,
    start: str = "2018-01-31",
    end: str = "2025-06-30",
) -> pd.DataFrame:
    """Build a panel of 8-K filing COUNTS per ticker per month.

    If `item_filter` is provided, count only filings whose `items` list
    contains at least one of the requested item codes (e.g. ['2.02']
    for earnings-only).
    """
    df = scored.dropna(subset=["filed_date"]).copy()
    if item_filter:
        # Parquet stores list-of-strings columns as numpy object arrays;
        # `(lst or [])` raises on numpy arrays because bool(array) is
        # ambiguous. Use an explicit length check + iter instead.
        item_set = set(item_filter)

        def _has_match(lst):
            if lst is None:
                return False
            try:
                if len(lst) == 0:
                    return False
            except TypeError:
                return False
            return any(it in item_set for it in lst)

        df = df[df["items"].apply(_has_match)]
    df["month_end"] = df["filed_date"].dt.normalize() + pd.offsets.MonthEnd(0)
    counts = (df.groupby(["month_end", "ticker"])
                .size()
                .unstack("ticker")
                .fillna(0))
    full_idx = pd.date_range(start, end, freq="ME")
    counts = counts.reindex(full_idx).fillna(0)
    return counts
