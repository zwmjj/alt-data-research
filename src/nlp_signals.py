"""Text extraction + Loughran-McDonald sentiment scoring of 10-K filings.

Pipeline:

  full-submission.txt  -->  extract_10k_text()  -->  raw 10-K text
                                                      |
                                                      v
                                                  lm_score()
                                                      |
                                                      v
                                  {pos_ratio, neg_ratio, unc_ratio,
                                   polarity, subjectivity, n_tokens}

We score **ratios** (count / total tokens) rather than raw counts so
a 100-page 10-K and a 50-page 10-Q are on the same scale.

The main factor we ship to downstream analysis is `neg_ratio - pos_ratio`
(net negativity) because Loughran-McDonald's famous finding is that
the *negative* word list is the stronger predictor of abnormal returns
— their positive list contains many words that are actually used in
litigation / risk-factor boilerplate and carry no real information.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pysentiment2 as ps

# Module-level singleton so we don't re-load the LM dictionary per-call.
_LM = None


def get_lm():
    global _LM
    if _LM is None:
        _LM = ps.LM()
    return _LM


# ────────────────────────────────────────────────────────────
# 1. Extract 10-K text from full-submission.txt
# ────────────────────────────────────────────────────────────
_SGML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_10K_DOC_RE = re.compile(
    r"<TYPE>10-K\s*<SEQUENCE>\s*\d+\s*<FILENAME>[^<]+\s*<DESCRIPTION>[^<]*\s*<TEXT>(.*?)</TEXT>",
    re.DOTALL,
)


def extract_10k_text(submission_path: Path) -> str:
    """Pull the primary 10-K document out of a full-submission SGML file.

    SEC full submissions embed multiple <DOCUMENT>...</DOCUMENT> blocks
    (exhibits, XBRL, graphics, etc.). We want just the 10-K itself
    — that's the block with `<TYPE>10-K`. Fall back to all-document
    concatenation if the regex doesn't match (some older filings).

    After extraction, we strip HTML/SGML tags and collapse whitespace
    so the LM tokenizer sees clean plain text.
    """
    try:
        raw = submission_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    # Preferred path: find just the 10-K document block.
    m = _10K_DOC_RE.search(raw)
    if m:
        body = m.group(1)
    else:
        # Fallback: strip all DOCUMENT wrappers, keep everything.
        body = raw

    # Strip HTML/SGML tags and decode common entities.
    body = _SGML_TAG_RE.sub(" ", body)
    body = (body
            .replace("&amp;", "&")
            .replace("&nbsp;", " ")
            .replace("&#39;", "'")
            .replace("&quot;", '"'))
    body = _WHITESPACE_RE.sub(" ", body).strip()
    return body


# ────────────────────────────────────────────────────────────
# 2. LM dictionary scoring
# ────────────────────────────────────────────────────────────
def lm_score(text: str) -> Dict[str, float]:
    """Apply Loughran-McDonald scoring to a body of text.

    Returns a dict with:
      n_tokens      — total tokens after LM tokenization/stemming
      pos_count     — raw positive-word count
      neg_count     — raw negative-word count
      pos_ratio     — pos_count / n_tokens
      neg_ratio     — neg_count / n_tokens
      net_neg       — neg_ratio - pos_ratio  (the headline factor)
      polarity      — (pos - neg) / (pos + neg + eps)
      subjectivity  — (pos + neg) / n_tokens

    If the text is empty or tokenization fails, returns all zeros
    with n_tokens=0 so downstream panels get a clean NaN mask.
    """
    if not text:
        return {k: 0.0 for k in
                ["n_tokens", "pos_count", "neg_count", "pos_ratio",
                 "neg_ratio", "net_neg", "polarity", "subjectivity"]}

    lm = get_lm()
    # LM's tokenize does lowercase + snowball stemming + remove stopwords
    tokens = lm.tokenize(text)
    n = len(tokens)
    if n == 0:
        return {"n_tokens": 0, "pos_count": 0, "neg_count": 0,
                "pos_ratio": 0.0, "neg_ratio": 0.0, "net_neg": 0.0,
                "polarity": 0.0, "subjectivity": 0.0}

    raw = lm.get_score(tokens)
    pos = int(raw["Positive"])
    neg = int(raw["Negative"])
    return {
        "n_tokens": int(n),
        "pos_count": pos,
        "neg_count": neg,
        "pos_ratio": pos / n,
        "neg_ratio": neg / n,
        "net_neg":   (neg - pos) / n,
        "polarity":  float(raw["Polarity"]),
        "subjectivity": float(raw["Subjectivity"]),
    }


# ────────────────────────────────────────────────────────────
# 3. Score an entire directory of 10-K filings
# ────────────────────────────────────────────────────────────
def score_filings_from_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Apply LM scoring to every row in a manifest built by
    `data_fetcher.list_10k_paths()`.

    Parameters
    ----------
    manifest : DataFrame with columns ['ticker', 'accession', 'path']

    Returns
    -------
    DataFrame adding columns from `lm_score` plus a derived
    `filed_date` parsed from the accession directory name.
    """
    rows = []
    for i, row in manifest.iterrows():
        text = extract_10k_text(Path(row["path"]))
        scores = lm_score(text)
        rows.append({**row.to_dict(), **scores})
        if (i + 1) % 25 == 0:
            print(f"  scored {i+1}/{len(manifest)} filings")
    scored = pd.DataFrame(rows)

    # Accession numbers look like 0000320193-23-000106 — the middle
    # 2-digit number is the year the filing was accepted, the suffix
    # is a sequential counter. We don't have the exact filed date
    # without a separate API call, so derive a coarse year from the
    # accession string and stamp to mid-year for now.
    def _infer_date(acc: str) -> Optional[pd.Timestamp]:
        try:
            yy = int(acc.split("-")[1])
            year = 2000 + yy if yy < 80 else 1900 + yy
            return pd.Timestamp(year=year, month=6, day=30)
        except Exception:
            return None

    scored["filed_date"] = scored["accession"].apply(_infer_date)
    return scored


# ────────────────────────────────────────────────────────────
# 4. Fetch precise filed dates from SEC EDGAR
# ────────────────────────────────────────────────────────────
def enrich_filed_dates(scored: pd.DataFrame, tickers_df: pd.DataFrame) -> pd.DataFrame:
    """Replace the coarse accession-derived dates with precise filed
    dates pulled from SEC EDGAR's submissions JSON endpoint.

    For each ticker, fetch `https://data.sec.gov/submissions/CIK{cik}.json`
    and look up each accession's filingDate.
    """
    import json
    import urllib.request
    ticker_cik = tickers_df.set_index("ticker")["cik"].to_dict()
    accession_to_date = {}

    for ticker in scored["ticker"].unique():
        cik = ticker_cik.get(ticker)
        if cik is None:
            continue
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Alt Data Research research@example.com"},
            )
            with urllib.request.urlopen(req) as r:
                data = json.loads(r.read().decode("utf-8"))
            recent = data.get("filings", {}).get("recent", {})
            acc_nums = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            for a, d in zip(acc_nums, filing_dates):
                accession_to_date[a] = pd.Timestamp(d)
        except Exception as e:
            print(f"  [warn] filed-date fetch failed for {ticker}: {str(e)[:80]}")

    # Accession numbers in our manifest use dashes, EDGAR JSON doesn't.
    def _lookup(acc):
        return accession_to_date.get(acc.replace("-", "")) or accession_to_date.get(acc)

    scored = scored.copy()
    precise = scored["accession"].apply(_lookup)
    # Keep the coarse date if precise lookup failed
    scored["filed_date"] = precise.fillna(scored["filed_date"])
    return scored
