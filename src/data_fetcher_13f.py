"""13F-HR institutional holdings — download + parse.

Pipeline:
  1. SEC name search → CIK for each top-tier filer (cached)
  2. sec-edgar-downloader pulls 13F-HR full-submission.txt files
  3. Parser strips XML <infoTable> blocks → DataFrame of holdings

The parser handles modern (post-2014) XML 13F format. Older
SGML-only filings exist but the SEC mandated XML schema in 2014, so
our 2018+ window doesn't need legacy parsing.
"""
from __future__ import annotations

import json
import pickle
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_13F = ROOT / "data" / "raw_13f"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW_13F.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

UA = "Alt Data Research research@example.com"


# ────────────────────────────────────────────────────────────
# 1. Filer CIK lookup
# ────────────────────────────────────────────────────────────
TOP_FILER_NAMES = [
    "Berkshire Hathaway", "Renaissance Technologies", "Bridgewater Associates",
    "Citadel Advisors", "Two Sigma Investments", "D E Shaw", "Millennium Management",
    "Point72", "Tiger Global", "Coatue Management", "Lone Pine Capital",
    "Viking Global", "Pershing Square", "Greenlight Capital", "Third Point",
    "Appaloosa", "Baupost", "AQR Capital", "Soros Fund", "Maverick Capital",
    "Adage Capital", "Marshall Wace", "Glenview Capital", "Egerton Capital",
    "Lansdowne Partners", "Paulson", "Elliott Management", "Tudor Investment",
    "Caxton", "Brevan Howard",
]


def lookup_filer_cik(name: str) -> Optional[str]:
    """Return the first 10-digit CIK matching `name` in EDGAR's
    company-search Atom feed, restricted to entities that have filed
    at least one 13F-HR.
    """
    q = name.replace(" ", "+")
    url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
           f"&company={q}&type=13F-HR&dateb=&owner=include&count=10&output=atom")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=10) as r:
            xml = r.read().decode("utf-8", errors="ignore")
        m = re.search(r"CIK=(\d{10})", xml)
        return m.group(1) if m else None
    except Exception:
        return None


def get_filer_ciks(refresh: bool = False) -> Dict[str, str]:
    """Resolve CIKs for the top-tier filer list. Cached as JSON."""
    cache = DATA_PROCESSED / "filers_13f.json"
    if cache.exists() and not refresh:
        with open(cache) as f:
            return json.load(f)
    out = {}
    for name in TOP_FILER_NAMES:
        cik = lookup_filer_cik(name)
        if cik:
            out[name] = cik
    with open(cache, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ────────────────────────────────────────────────────────────
# 2. Download 13F filings
# ────────────────────────────────────────────────────────────
def download_13f_filings(
    filer_ciks: Dict[str, str],
    after: str = "2018-01-01",
    before: str = "2024-12-31",
) -> int:
    """Pull every 13F-HR filing for each filer in the input dict."""
    from sec_edgar_downloader import Downloader
    dl = Downloader("Alt Data Research", "research@example.com", str(DATA_RAW_13F))
    n = 0
    for i, (name, cik) in enumerate(filer_ciks.items()):
        try:
            count = dl.get("13F-HR", cik, after=after, before=before, download_details=False)
            n += count
            print(f"  [{i+1:2d}/{len(filer_ciks)}] {name:32s} CIK={cik}: {count} filings (cum={n})")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(filer_ciks)}] {name:32s} ERROR: {str(e)[:80]}")
    return n


# ────────────────────────────────────────────────────────────
# 3. Parse 13F-HR XML to extract holdings
# ────────────────────────────────────────────────────────────
_INFOTABLE_RE = re.compile(r"<infoTable>(.*?)</infoTable>", re.DOTALL | re.IGNORECASE)
_FIELD_RE = {
    "name":  re.compile(r"<nameOfIssuer>(.*?)</nameOfIssuer>", re.DOTALL | re.IGNORECASE),
    "class": re.compile(r"<titleOfClass>(.*?)</titleOfClass>", re.DOTALL | re.IGNORECASE),
    "cusip": re.compile(r"<cusip>(.*?)</cusip>", re.DOTALL | re.IGNORECASE),
    "value": re.compile(r"<value>(.*?)</value>", re.DOTALL | re.IGNORECASE),
    "shrs":  re.compile(r"<sshPrnamt>(.*?)</sshPrnamt>", re.DOTALL | re.IGNORECASE),
    "type":  re.compile(r"<sshPrnamtType>(.*?)</sshPrnamtType>", re.DOTALL | re.IGNORECASE),
}


def parse_13f_filing(submission_path: Path) -> pd.DataFrame:
    """Parse a 13F-HR full-submission.txt into a DataFrame of holdings.

    Returns columns: name, class, cusip, value (as filed), shrs, type.
    Empty DataFrame if no infoTable blocks found (some pre-2014 filings,
    or amendments / transitional filings).
    """
    try:
        raw = submission_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame()

    blocks = _INFOTABLE_RE.findall(raw)
    if not blocks:
        return pd.DataFrame()

    rows = []
    for block in blocks:
        row = {}
        for k, rx in _FIELD_RE.items():
            m = rx.search(block)
            row[k] = m.group(1).strip() if m else None
        rows.append(row)

    df = pd.DataFrame(rows)
    # Numeric coercion — value is sometimes in thousands (pre-2023 SEC
    # rules) and sometimes in dollars (2023+). We don't normalize here;
    # downstream factor logic uses *changes* in n_holders or
    # percentage changes in value, both of which are unit-invariant.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["shrs"] = pd.to_numeric(df["shrs"], errors="coerce")
    return df


# ────────────────────────────────────────────────────────────
# 4. Build holdings panel: list every (filer, period_of_report, holding)
# ────────────────────────────────────────────────────────────
def build_holdings_panel(filer_ciks: Dict[str, str]) -> pd.DataFrame:
    """Walk every downloaded 13F-HR and concatenate parsed rows.

    Returns long-format DataFrame:
      filer_name | filer_cik | accession | period_end | name | class |
      cusip | value | shrs | type
    """
    cache = DATA_PROCESSED / "holdings_13f_long.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    rows = []
    base = DATA_RAW_13F / "sec-edgar-filings"
    for filer_name, cik in filer_ciks.items():
        cik_dir = base / cik
        if not cik_dir.exists():
            continue
        form_dir = cik_dir / "13F-HR"
        if not form_dir.exists():
            continue
        for accession_dir in form_dir.iterdir():
            if not accession_dir.is_dir():
                continue
            sub = accession_dir / "full-submission.txt"
            if not sub.exists():
                continue
            df = parse_13f_filing(sub)
            if df.empty:
                continue

            period_end = _extract_period_of_report(sub)
            df["filer_name"] = filer_name
            df["filer_cik"] = cik
            df["accession"] = accession_dir.name
            df["period_end"] = period_end
            rows.append(df)

    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(cache)
    return panel


def _extract_period_of_report(submission_path: Path) -> Optional[pd.Timestamp]:
    """Pull <PERIOD>YYYYMMDD</PERIOD> or <periodOfReport>YYYY-MM-DD</periodOfReport>."""
    try:
        head = submission_path.read_text(encoding="utf-8", errors="ignore")[:20000]
    except Exception:
        return None
    # Two formats: SGML header `<PERIOD>YYYYMMDD` or XML `<periodOfReport>YYYY-MM-DD`
    m = re.search(r"<PERIOD>(\d{8})", head)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    m = re.search(r"<periodOfReport>([\d\-]+)", head)
    if m:
        return pd.to_datetime(m.group(1))
    return None


# ────────────────────────────────────────────────────────────
# 5. CUSIP → ticker mapping (S&P 100 universe via WRDS CRSP)
# ────────────────────────────────────────────────────────────
def get_cusip_ticker_map(tickers: List[str]) -> Dict[str, str]:
    """Resolve CUSIP for each input ticker via CRSP `crsp.stocknames`.

    Returns CUSIP→ticker mapping. CRSP gives 8-digit CUSIPs; 13F holdings
    report 9-digit CUSIPs (8 + check digit), so we match on the leading 8.
    """
    cache = DATA_PROCESSED / f"cusip_ticker_map_{len(tickers)}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Load WRDS creds from ~/quant/.env
    import os
    env_path = Path.home() / "quant" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    import wrds
    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    try:
        tk_list = "','".join(tickers)
        # crsp.stocknames uses `nameenddt` (not `nameendt`) for the
        # last date a name applied. Restricting to >=2018 keeps us on
        # the modern CUSIP for tickers that have been re-issued
        # (e.g. META vs FB pre-2022 rename).
        query = f"""
            SELECT DISTINCT cusip, ticker, comnam
            FROM   crsp.stocknames
            WHERE  ticker IN ('{tk_list}')
              AND  nameenddt >= '2018-01-01'
        """
        df = db.raw_sql(query)
    finally:
        db.close()

    # CRSP CUSIPs are 8-digit. Build a map from 8-digit CUSIP to ticker.
    out = {}
    for _, row in df.iterrows():
        c = str(row["cusip"]).strip().upper()
        if c and c != "NAN":
            out[c[:8]] = row["ticker"]

    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"[data] CUSIP map: {len(out)} unique CUSIPs across {df['ticker'].nunique()} tickers")
    return out
