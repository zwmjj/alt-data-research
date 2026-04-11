"""Parallel 8-K downloader using SEC EDGAR's submissions API directly.

`sec-edgar-downloader 5.x` runs at ~0.5 filings/sec — SEC actually
allows 10 req/s. This module hits SEC EDGAR concurrently with proper
rate limiting at ~8 req/s aggregate (leaving headroom under the 10/s
limit) and writes files in the same path structure as
`sec-edgar-downloader` so `data_fetcher_8k.list_8k_paths()` can pick
them up without modification.

Two-phase approach:
  1. For each ticker → fetch submissions JSON to get accession list
     (1 request per ticker, fast)
  2. For each (ticker, accession) → fetch the full submission file
     into ./data/raw_8k/sec-edgar-filings/{TICKER}/8-K/{ACCESSION}/full-submission.txt

Throttling: a global token-bucket limiter shared across all worker
threads ensures we never exceed the 10/sec ceiling.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_8K = ROOT / "data" / "raw_8k" / "sec-edgar-filings"
DATA_RAW_8K.mkdir(parents=True, exist_ok=True)

UA = "Alt Data Research research@example.com"


# ────────────────────────────────────────────────────────────
# Token-bucket rate limiter
# ────────────────────────────────────────────────────────────
class RateLimiter:
    """Simple thread-safe token-bucket. Default 8 tokens/sec."""

    def __init__(self, rate_per_sec: float = 8.0):
        self.rate = rate_per_sec
        self.tokens = rate_per_sec
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens < 1:
                wait = (1 - self.tokens) / self.rate
                time.sleep(wait)
                self.tokens = 0
                self.last = time.monotonic()
            else:
                self.tokens -= 1


_LIMITER = RateLimiter(rate_per_sec=8.0)


def _http_get(url: str, retries: int = 3) -> bytes:
    """Rate-limited GET with retry on 429/5xx."""
    for attempt in range(retries):
        _LIMITER.acquire()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"GET failed after {retries} retries: {url}")


# ────────────────────────────────────────────────────────────
# Submissions API → accession list
# ────────────────────────────────────────────────────────────
def fetch_8k_accessions(cik: str, after: str, before: str) -> List[Dict]:
    """Return a list of {accession, filed_date, primary_doc} for all
    8-K filings for a CIK in the date range.

    Uses SEC EDGAR submissions API:
        https://data.sec.gov/submissions/CIK{cik}.json

    The recent filings array has up to ~1000 entries; older filings
    are in `files.recent.older` JSON references that we don't need
    for our 2018-2024 window.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    raw = _http_get(url)
    data = json.loads(raw.decode("utf-8"))

    rec = data.get("filings", {}).get("recent", {})
    forms = rec.get("form", [])
    dates = rec.get("filingDate", [])
    accs = rec.get("accessionNumber", [])
    primary = rec.get("primaryDocument", [])

    out = []
    for form, date, acc, doc in zip(forms, dates, accs, primary):
        if form != "8-K":
            continue
        if date < after or date > before:
            continue
        out.append({"accession": acc, "filed_date": date, "primary_doc": doc})
    return out


def fetch_full_submission(cik: str, accession: str) -> bytes:
    """Pull the full SGML/XML submission file for a single filing.

    Path: https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{acc}-index.htm
    But the actual content is at:
          https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{acc}.txt
    """
    cik_int = int(cik)
    acc_no_dashes = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{accession}.txt"
    return _http_get(url)


# ────────────────────────────────────────────────────────────
# Concurrent downloader
# ────────────────────────────────────────────────────────────
def download_for_ticker(
    ticker: str,
    cik: str,
    after: str,
    before: str,
    skip_existing: bool = True,
) -> int:
    """Download all 8-K filings for one ticker. Returns count downloaded
    (excluding skipped existing files).
    """
    accs = fetch_8k_accessions(cik, after, before)
    n_new = 0
    for entry in accs:
        accession = entry["accession"]
        out_dir = DATA_RAW_8K / ticker / "8-K" / accession
        out_path = out_dir / "full-submission.txt"
        if skip_existing and out_path.exists():
            continue
        try:
            content = fetch_full_submission(cik, accession)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)
            n_new += 1
        except Exception as e:
            print(f"  [{ticker}] {accession}: ERROR {str(e)[:60]}")
    return n_new


def download_8k_parallel(
    tickers_with_cik: pd.DataFrame,
    after: str = "2018-01-01",
    before: str = "2024-12-31",
    max_workers: int = 6,
) -> int:
    """Concurrently download 8-Ks for many tickers.

    Parameters
    ----------
    tickers_with_cik : DataFrame with columns 'ticker' and 'cik'
        Output of `data_fetcher.get_sp100_tickers()` works directly.
    """
    total = 0
    t0 = time.time()

    def _work(row):
        ticker = row.ticker
        cik = row.cik
        try:
            return ticker, download_for_ticker(ticker, cik, after, before)
        except Exception as e:
            return ticker, f"ERROR: {str(e)[:80]}"

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, row) for row in tickers_with_cik.itertuples(index=False)]
        completed = 0
        for fut in as_completed(futures):
            ticker, result = fut.result()
            completed += 1
            if isinstance(result, int):
                total += result
                if completed % 5 == 0 or completed == len(futures):
                    elapsed = time.time() - t0
                    rate = total / elapsed if elapsed > 0 else 0
                    print(f"  [{completed}/{len(futures)}] {ticker}: +{result}  "
                          f"(cum={total}, {elapsed/60:.1f}min, {rate:.1f} filings/sec)")
            else:
                print(f"  [{completed}/{len(futures)}] {ticker}: {result}")

    elapsed = time.time() - t0
    print(f"\n[done] {total} new filings in {elapsed/60:.1f} min ({total/elapsed:.1f}/s avg)")
    return total
