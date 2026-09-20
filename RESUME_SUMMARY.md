# Alternative Data Alpha Research — One-Page Summary

> **Combining three SEC-derived signals — 10-K Loughran-McDonald
> sentiment, 13F holdings flow, 8-K event sentiment — produces a
> long-short factor with ICIR 0.80 and Sharpe 0.75 on S&P 100 over
> 2018-2024. The sign of each component was fitted on the full sample,
> so the accompanying t-stat of 2.11 is an in-sample fit statistic, not
> a significance test.**

## Headline result

| Factor                  | ICIR     | t-stat  | L/S Sharpe | L/S ann. |
|-------------------------|----------|---------|------------|----------|
| Combined alt-data       | +0.80    | +2.11   | +0.75      | +7.7%    |
| 10-K LM sentiment       | +0.38    | +1.00   | +0.54      | +7.2%    |
| 13F Δ shares (3m fwd)   | −0.73    | −1.86   | −0.53*     | —        |
| 8-K LM sentiment        | +0.42    | +1.10   | +0.23      | +3.0%    |
| 12-month momentum       | +0.03    | +0.08   | −0.001     | −0.0%    |
| Low-volatility          | −0.66    | −1.62   | −0.71      | −15.0%   |

\* Task 2's strongest signal is the 3-month-forward 13F delta, which
   has *negative* IC (the "13F lag" finding — institutional buys are
   already priced in by the time the filing is public). For Task 4
   blending, all three components are sign-flipped to their full-sample
   empirical direction; all three were negative before the flip.

**The row that matters is the last one.** Low-volatility is shown at
its textbook sign while the alt-data rows are shown at their
best-fitting sign. Put it through the same rule and it becomes ICIR
+0.66, t +1.62, L/S Sharpe +0.71, annualized **+15.0%** — a lower ICIR
than the combined factor but twice the return. This repository
previously claimed the alt-data block outperformed every traditional
price-based factor; that claim came from the asymmetry, and has been
withdrawn.

## What this project demonstrates

| Capability                                                | Evidence                                                                       |
|-----------------------------------------------------------|--------------------------------------------------------------------------------|
| **Multi-source SEC EDGAR pipeline**                       | 8,613 filings parsed across 10-K (687), 13F-HR (672), 8-K (7,254)              |
| **NLP factor construction**                               | Loughran-McDonald dictionary scoring, regex SGML parsing, EX-99 extraction     |
| **Custom rate-limited downloader**                        | `fast_8k_downloader.py` — 10× speedup over `sec-edgar-downloader` via threadpool |
| **WRDS CRSP integration**                                 | CUSIP→permno→ticker mapping via `crsp.stocknames`, monthly returns join         |
| **Standard factor research methodology**                  | Spearman IC, ICIR, t-stat, hit rate, quintile L/S, multi-horizon (1m/3m/6m)    |
| **Cross-factor correlation analysis**                     | 8×8 correlation matrices showing 3 alt-data axes are nearly orthogonal         |
| **Statistical significance testing**                      | Spearman IC t-stats reported throughout; Task 4's t = 2.11 documented as sign-contaminated rather than presented as significance |
| **Honest negative findings**                              | All three components have negative raw IC; the write-up says so and withdraws the superseded outperformance claim |
| **Reproducibility**                                       | One-command per task, parquet caching, deterministic                            |

## The interesting math

Three components with similar individual ICIR (0.38–0.49) and slightly
**negative** pairwise correlations (the 3 alt-data factors are
essentially independent at \|corr\| < 0.18). Diversification math:

- Theoretical 1/√3 multiplier for 3 independent signals: **1.73×**
- Actual realized multiplier: **0.80 / 0.43 = 1.86×**

The 0.13 excess over theory comes from the components being slightly
*better* than independent — Task 1 ⊥ Task 2 = −0.030, Task 2 ⊥ Task 3
= −0.003. Negative correlations diversify better than zero-correlated
signals.

## What happened to traditional factors in 2018-2024

The S&P 100 from 2018-2024 was dominated by mega-cap tech. At textbook
signs, `mom12` had cross-sectional Sharpe ≈ 0, `mom1_reversal` −0.20 and
`lowvol` −0.71, with cumulative L/S returns of −11%, −25% and −65%.

Calling that a collapse is the wrong reading. A long-short factor's sign
is a convention, and this project did not treat it as fixed for its own
signals. Under a like-for-like rule, the flipped low-vol factor — long
high-volatility names, short low-volatility ones — earned +0.71 Sharpe
and +15.0% a year on exactly the same universe and window. The mega-cap
growth rally was not the absence of a price-based factor; it was the
payoff to one.

The claim that survives is narrower and is the one worth making: the
combined alt-data factor's correlation with the traditional stack is
under |0.12| in every pair, and correlation is invariant to sign. It is
additive to an existing multi-factor book rather than a repackaging of
it — independently of whether its in-sample alpha holds up.

## Tech stack

- **Python 3.13** (numpy, pandas 2.2, pyarrow, matplotlib)
- **`pysentiment2`** for Loughran-McDonald dictionary
- **`sec-edgar-downloader`** + custom threadpool downloader
- **`wrds`** + WRDS CRSP for CUSIP mapping & clean monthly returns
- **`pandas-datareader`** for FRED macro
- **`transformers` + `torch` (CUDA)** — FinBERT implementation shipped
  but blocked on this environment by torchvision/torch nightly version
  mismatch (documented in README)

## Repo structure

~2,500 LOC across 10 Python modules:

```
src/
├── data_fetcher.py             10-K + S&P 100 + WRDS returns
├── data_fetcher_13f.py         13F filer search + parser + CUSIP map
├── data_fetcher_8k.py          8-K parser w/ item codes + EX-99
├── fast_8k_downloader.py       threadpool downloader (10× speedup)
├── nlp_signals.py              LM dictionary scoring
├── factor_analysis.py          IC/ICIR/quintile/correlation toolkit
├── factor_13f.py               holdings panels + delta signals
├── factor_8k_news.py           8-K LM + FinBERT (impl ready)
├── factor_combined.py          sign-flip + z-score + blend
└── run_task{1,2,3,4}.py        end-to-end runners
```

19 reference figures in `results/figures/`. 12 summary CSVs in
`results/`. Full narrative in `README.md` (~700 lines).

## Limitations & next steps

1. **In-sample sign selection** (the binding one): `factor_combined.py`
   orients each component by its full-sample IC, then a t-stat is
   computed on that same sample. Three sign choices plus a blending
   choice were fitted to 83 observations. The result is an in-sample fit,
   not a test. Fixing signs on a training window and reporting only the
   held-out period is the required next experiment; it has not been run.
2. **Benchmark comparison was not like-for-like**: alt-data signals were
   sign-selected, traditional factors were not. Corrected, flipped
   `lowvol` gives +0.71 Sharpe and +15.0% annualized versus the combined
   factor's +0.75 and +7.7%.
3. **Sample size**: 83 monthly observations. WRDS CRSP full universe
   (3,000 names × 25 years) would tighten the standard error — though a
   larger sample makes a fitted sign more precise, not more valid.
4. **Single regime (2018-2024)**. A cross-regime test (2010-2017 vs
   2018-2024) is the natural control.
5. **FinBERT not run**: torchvision/torch nightly mismatch documented;
   implementation is in `factor_8k_news.score_filings_finbert` ready
   to ship on a clean install.
6. **Universe homogeneity**: All 3 components share S&P 100. The
   combined factor's IC would likely strengthen on a small/mid-cap
   universe with more idiosyncratic event flow and less price
   discovery.

---

## Related open-source repos

This project is one of six interconnected repositories that together
form a complete quantitative research platform. The alt-data work
builds on the shared `kuant-core` library and shares its
reproducibility protocol with `kuant-research`.

| Repo | Role | LOC |
|---|---|---|
| **[alt-data-research](https://github.com/zwmjj/alt-data-research)** | This project — SEC NLP + 13F flow alpha pipeline; combined ICIR 0.80 in-sample, signs fitted on the same sample | ~2.5k |
| **[kuant-research](https://github.com/zwmjj/kuant-research)** | 14 reproducible empirical studies with committed expected outputs + reproducibility gate | ~3k |
| **[kuant-core](https://github.com/zwmjj/kuant-core)** | Production quant research library — event-driven backtester, 28+ factor library, walk-forward CV, multi-market data (US CRSP + China A-share), cost model, risk toolkit | ~20k |
| **[kuant-strategies](https://github.com/zwmjj/kuant-strategies)** | 25+ trading strategies built on `kuant-core`: momentum, mean-reversion, cross-asset, crypto, options, ML, alt-data | ~17k |
| **[kuant-api](https://github.com/zwmjj/kuant-api)** | FastAPI research backend — 20 routers serving backtests, factor research, Monaco code IDE, WebSocket monitoring, SOP gate-check dashboard | ~5k |
| **[kuant-web](https://github.com/zwmjj/kuant-web)** | Next.js 16 + Tailwind + Recharts frontend — 20 panels for interactive research, factor analysis, live monitoring, multi-agent control | ~7k |

**Total public code: ~55,000 LOC across 6 MIT-licensed repositories.**

### How a recruiter should read these

- **`alt-data-research`** (this repo) — the *NLP / alt-data* story:
  a multi-source SEC pipeline, and a worked example of catching a
  methodology error in one's own headline. The combined factor's
  significance claim was withdrawn after the in-sample sign selection
  was identified; the write-up now shows the corrected like-for-like
  comparison instead.
- **`kuant-research`** — demonstrates *reproducibility discipline*.
  Every study ships with `expected_output.json` and a `run.py` that
  re-runs the analysis end-to-end from a fresh clone.
- **`kuant-core`** — demonstrates *library engineering*. Clean
  public API, multi-market data plumbing, 5 cost models, SOP-compliant
  gate-check system.
- **`kuant-strategies`** — demonstrates *breadth*. 25 strategies
  across 7 categories and 4 asset classes, all built on the same
  underlying engine.
- **`kuant-api` + `kuant-web`** — demonstrate *full-stack depth*
  for quant-dev / quant-infra positions. Not strictly necessary for
  pure research roles.

---

**Repo:** https://github.com/zwmjj/alt-data-research
