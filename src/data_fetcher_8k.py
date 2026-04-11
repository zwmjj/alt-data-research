"""8-K filings — download + parse.

8-K is the SEC form companies file when they have a "material event" to
disclose: earnings releases, M&A announcements, executive departures,
restructuring, audit-firm changes, etc. SEC requires 8-K within 4
business days of the triggering event, so 8-Ks are the closest thing
to "real news flow" available in fully-public, free, structured data.

Each 8-K declares one or more `Item NN.NN` codes that classify the
event. The most common codes:

  1.01  Entry into a Material Definitive Agreement
  2.02  Results of Operations and Financial Condition (earnings)
  5.02  Departure / Election / Compensation of Officers / Directors
  7.01  Regulation FD Disclosure
  8.01  Other Events
  9.01  Financial Statements and Exhibits

This module:
  1. Walks DATA_RAW_8K/sec-edgar-filings to build a manifest
  2. Parses each filing's full-submission.txt to extract:
       - filed_date (from <ACCEPTANCE-DATETIME>)
       - period_of_report
       - item codes
       - main 8-K narrative text
       - press-release exhibit text (Item 2.02 earnings releases ship the
         press release as an EX-99.1 exhibit and that's where the
         interesting language is)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_8K = ROOT / "data" / "raw_8k"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW_8K.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────
# Discovery
# ────────────────────────────────────────────────────────────
def list_8k_paths() -> pd.DataFrame:
    """Walk the 8-K download tree and return [ticker, accession, path]."""
    base = DATA_RAW_8K / "sec-edgar-filings"
    if not base.exists():
        return pd.DataFrame(columns=["ticker", "accession", "path"])
    rows = []
    for ticker_dir in base.iterdir():
        if not ticker_dir.is_dir():
            continue
        form_dir = ticker_dir / "8-K"
        if not form_dir.exists():
            continue
        for accession_dir in form_dir.iterdir():
            if not accession_dir.is_dir():
                continue
            sub = accession_dir / "full-submission.txt"
            if sub.exists():
                rows.append({
                    "ticker": ticker_dir.name,
                    "accession": accession_dir.name,
                    "path": str(sub),
                })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# Parsing
# ────────────────────────────────────────────────────────────
_SGML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ACCEPT_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
_PERIOD_RE = re.compile(r"<PERIOD>(\d{8})")

# Item codes look like "Item 2.02" or "ITEM 2.02" — pull the numeric part.
_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}\.\d{2})", re.IGNORECASE)

# Pull the body of the 8-K's primary document (the form itself, not exhibits).
# We match the FIRST <DOCUMENT> block whose TYPE is 8-K.
_8K_DOC_RE = re.compile(
    r"<TYPE>8-K\s*<SEQUENCE>\s*\d+\s*<FILENAME>[^\n]+\s*(?:<DESCRIPTION>[^\n]*\s*)?<TEXT>(.*?)</TEXT>",
    re.DOTALL | re.IGNORECASE,
)

# EX-99.1 is the most common press-release exhibit type. We pull it
# separately because Item 2.02 earnings releases bury the actual sentiment
# in the EX-99.1 narrative, not in the bare 8-K shell.
_EX_99_RE = re.compile(
    r"<TYPE>EX-99(?:\.\d+)?\s*<SEQUENCE>\s*\d+\s*<FILENAME>[^\n]+\s*(?:<DESCRIPTION>[^\n]*\s*)?<TEXT>(.*?)</TEXT>",
    re.DOTALL | re.IGNORECASE,
)


def _strip(text: str) -> str:
    text = _SGML_TAG_RE.sub(" ", text)
    text = (text
            .replace("&amp;", "&")
            .replace("&nbsp;", " ")
            .replace("&#39;", "'")
            .replace("&quot;", '"'))
    return _WS_RE.sub(" ", text).strip()


def parse_8k_filing(submission_path: Path) -> dict:
    """Extract metadata + text from a single 8-K full-submission file.

    Returns a dict with:
      filed_date  — from ACCEPTANCE-DATETIME header
      period_end  — from PERIOD header (if present)
      items       — list of Item codes (e.g. ['2.02', '9.01'])
      n_items     — len(items)
      body        — cleaned 8-K narrative text
      ex99        — cleaned press-release exhibit text (often empty)
      n_chars     — len(body) + len(ex99)
    """
    out = {
        "filed_date": None, "period_end": None, "items": [], "n_items": 0,
        "body": "", "ex99": "", "n_chars": 0,
    }
    try:
        raw = submission_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out

    # Header dates
    head = raw[:4000]
    m = _ACCEPT_RE.search(head)
    if m:
        try:
            out["filed_date"] = pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S")
        except Exception:
            pass
    m = _PERIOD_RE.search(head)
    if m:
        try:
            out["period_end"] = pd.to_datetime(m.group(1), format="%Y%m%d")
        except Exception:
            pass

    # Body (primary 8-K document)
    m = _8K_DOC_RE.search(raw)
    body_raw = m.group(1) if m else raw[:200000]  # fallback to first 200k chars
    out["body"] = _strip(body_raw)

    # Item codes — search the cleaned body so we don't pick up tag text.
    items = sorted(set(_ITEM_RE.findall(out["body"])))
    out["items"] = items
    out["n_items"] = len(items)

    # Press release exhibit (concatenate all EX-99.x if multiple)
    ex_chunks = _EX_99_RE.findall(raw)
    if ex_chunks:
        out["ex99"] = _strip(" ".join(ex_chunks))

    out["n_chars"] = len(out["body"]) + len(out["ex99"])
    return out


def parse_all_filings(manifest: pd.DataFrame) -> pd.DataFrame:
    """Apply parse_8k_filing to every row in the manifest. Returns a
    DataFrame with one row per filing plus the manifest columns.
    """
    rows = []
    for i, row in manifest.iterrows():
        meta = parse_8k_filing(Path(row["path"]))
        rows.append({**row.to_dict(), **meta})
        if (i + 1) % 200 == 0:
            print(f"  parsed {i+1}/{len(manifest)} filings")
    return pd.DataFrame(rows)
