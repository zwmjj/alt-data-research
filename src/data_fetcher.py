"""Data acquisition: S&P 100 tickers, SEC 10-K filings, monthly returns.

Three sources:
  - Wikipedia for the S&P 100 constituent list (free, one HTML page)
  - SEC EDGAR for 10-K filings (free, rate-limited to 10 req/s with
    a User-Agent header)
  - WRDS CRSP for monthly returns (primary), yfinance fallback if
    WRDS creds are missing
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────
# 1. S&P 100 constituents
# ────────────────────────────────────────────────────────────
def get_sp100_tickers(refresh: bool = False) -> pd.DataFrame:
    """Return current S&P 100 constituents (ticker, name, cik).

    Scrapes Wikipedia — the page includes ticker, company name, and
    sector. We enrich with CIK numbers from the SEC company_tickers
    JSON so the downstream EDGAR downloader doesn't need to resolve
    tickers at fetch time.
    """
    cache = DATA_PROCESSED / "sp100_tickers.pkl"
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Wikipedia S&P 100 table — the 2nd table on the page has the
    # current constituents. Use io.StringIO to avoid FutureWarning.
    import urllib.request
    import io
    req = urllib.request.Request(
        "https://en.wikipedia.org/wiki/S%26P_100",
        headers={"User-Agent": "Mozilla/5.0 alt-data-research"},
    )
    with urllib.request.urlopen(req) as r:
        html = r.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    # Pick the table that has a Symbol column and ~100 rows
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c for c in cols) and 90 <= len(t) <= 110:
            df = t.copy()
            break
    else:
        raise RuntimeError("Could not find S&P 100 table on Wikipedia")

    # Normalize column names
    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if "symbol" in lc:
            rename[c] = "ticker"
        elif "name" in lc:
            rename[c] = "name"
        elif "sector" in lc:
            rename[c] = "sector"
    df = df.rename(columns=rename)[["ticker", "name", "sector"]]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B

    # SEC tickers JSON — maps ticker -> CIK
    req = urllib.request.Request(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "Alt Data Research research@example.com"},
    )
    import json
    with urllib.request.urlopen(req) as r:
        sec_tickers = json.loads(r.read().decode("utf-8"))

    ticker_to_cik = {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in sec_tickers.values()
    }
    df["cik"] = df["ticker"].map(ticker_to_cik)
    df = df.dropna(subset=["cik"]).reset_index(drop=True)

    with open(cache, "wb") as f:
        pickle.dump(df, f)
    print(f"[data] S&P 100: {len(df)} tickers with CIK")
    return df


# ────────────────────────────────────────────────────────────
# 2. SEC 10-K filings
# ────────────────────────────────────────────────────────────
def download_10k_filings(
    tickers: List[str],
    after: str = "2014-01-01",
    before: str = "2024-12-31",
    email: str = "research@example.com",
) -> int:
    """Download 10-K filings for a list of tickers via sec-edgar-downloader.

    The downloader handles SEC rate limiting (10 req/s), retries, and
    respects SEC's preferred User-Agent format.

    Returns number of filings downloaded.
    """
    from sec_edgar_downloader import Downloader
    dl = Downloader("Alt Data Research", email, str(DATA_RAW))
    n = 0
    for i, tk in enumerate(tickers):
        try:
            count = dl.get(
                "10-K", tk, after=after, before=before, download_details=False,
            )
            n += count
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(tickers)}] {tk}: +{count} filings (total {n})")
        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {tk}: ERROR {str(e)[:100]}")
    return n


def list_10k_paths() -> pd.DataFrame:
    """Scan DATA_RAW/sec-edgar-filings/ for all downloaded 10-K files.

    Returns a DataFrame with [ticker, accession, filed_date, path].
    filed_date is parsed from the accession number directory name.
    """
    base = DATA_RAW / "sec-edgar-filings"
    if not base.exists():
        return pd.DataFrame(columns=["ticker", "accession", "path"])
    rows = []
    for ticker_dir in base.iterdir():
        if not ticker_dir.is_dir():
            continue
        form_dir = ticker_dir / "10-K"
        if not form_dir.exists():
            continue
        for accession_dir in form_dir.iterdir():
            if not accession_dir.is_dir():
                continue
            # Primary filing is usually full-submission.txt
            submission = accession_dir / "full-submission.txt"
            if submission.exists():
                rows.append({
                    "ticker": ticker_dir.name,
                    "accession": accession_dir.name,
                    "path": str(submission),
                })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# 3. Monthly returns — WRDS primary, yfinance fallback
# ────────────────────────────────────────────────────────────
def fetch_returns_wrds(
    tickers: List[str],
    start: str = "2014-01-01",
    end: str = "2025-12-31",
) -> Optional[pd.DataFrame]:
    """Pull monthly CRSP returns for a set of tickers.

    Uses `msenames` to resolve tickers -> permno, then `msf` for the
    return series. Returns a wide DataFrame (index=month-end,
    columns=ticker, values=return decimal), or None if WRDS creds
    are not available.
    """
    cache = DATA_PROCESSED / f"wrds_returns_{start}_{end}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Load WRDS creds from ~/quant/.env (colocated with the main Kuant repo)
    env_path = Path.home() / "quant" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    if not (os.environ.get("WRDS_USERNAME") and os.environ.get("WRDS_PASSWORD")):
        print("[data] WRDS creds not found, skipping WRDS")
        return None

    try:
        import wrds
    except ImportError:
        print("[data] wrds package not installed")
        return None

    try:
        db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    except Exception as e:
        print(f"[data] WRDS connection failed: {e}")
        return None

    try:
        # Resolve tickers -> permnos via msenames
        tk_list = "','".join(tickers)
        query = f"""
            SELECT DISTINCT permno, ticker
            FROM   crsp.msenames
            WHERE  ticker IN ('{tk_list}')
              AND  nameendt >= '{start}'
        """
        names = db.raw_sql(query)
        print(f"[data] WRDS resolved {names['permno'].nunique()} permnos for "
              f"{len(tickers)} tickers")

        if len(names) == 0:
            return None

        permno_list = ",".join(str(int(p)) for p in names["permno"].unique())
        query = f"""
            SELECT date, permno, ret
            FROM   crsp.msf
            WHERE  permno IN ({permno_list})
              AND  date BETWEEN '{start}' AND '{end}'
              AND  ret IS NOT NULL
        """
        ret = db.raw_sql(query)
    finally:
        db.close()

    # Merge permno → ticker (prefer most recent name)
    ret = ret.merge(names.drop_duplicates("permno"), on="permno", how="left")
    ret["date"] = pd.to_datetime(ret["date"]) + pd.offsets.MonthEnd(0)
    wide = ret.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")

    with open(cache, "wb") as f:
        pickle.dump(wide, f)
    print(f"[data] WRDS returns: {wide.shape[0]} months x {wide.shape[1]} tickers")
    return wide


def fetch_returns_yfinance(
    tickers: List[str],
    start: str = "2014-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Monthly total returns from yfinance. Fallback path."""
    cache = DATA_PROCESSED / f"yf_returns_{start}_{end}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker")
    # When multiple tickers, raw has a MultiIndex column. Extract Close.
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)})
    else:
        close = raw["Close"].to_frame(tickers[0]) if "Close" in raw.columns else raw
    monthly = close.resample("ME").last().pct_change().dropna(how="all")
    with open(cache, "wb") as f:
        pickle.dump(monthly, f)
    print(f"[data] yfinance returns: {monthly.shape[0]} months x {monthly.shape[1]} tickers")
    return monthly


def fetch_returns(tickers, start="2014-01-01", end="2025-12-31") -> pd.DataFrame:
    """Try WRDS first, fall back to yfinance."""
    ret = fetch_returns_wrds(tickers, start, end)
    if ret is not None and not ret.empty:
        return ret
    return fetch_returns_yfinance(tickers, start, end)
