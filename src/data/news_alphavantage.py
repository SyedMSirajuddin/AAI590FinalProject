"""
Alpha Vantage news-and-sentiment ingestion.

Primary news source for the project. Replaces the earlier Finnhub/GDELT
fetchers. Produces the SAME record schema the rest of the pipeline expects,
plus two extra columns carrying Alpha Vantage's own sentiment:

    [published_at (naive UTC), ticker, headline, summary, source, url,
     av_sentiment_score, av_relevance]

The two extra columns let the project benchmark its own FinBERT scores
against an independent commercial scorer (see Notebook 06); they are ignored
by clean_news() and every downstream stage, which key only on the shared
columns.

Design notes carried over from field testing against the live API:
  * Alphabet is tagged under GOOG, not GOOGL, so a project ticker may map to
    a different Alpha Vantage query symbol (AV_SYMBOL).
  * The DOC feed caps a response at 1000 articles, so multi-year history is
    paginated forward by advancing `time_from` past the newest article seen.
  * A malformed advancing cursor is rejected by the API with "Invalid
    inputs" and silently truncates a ticker; the cursor is therefore
    formatted to strict minute precision with a guarded parse, and the
    fetcher returns a `complete` flag so truncated pulls are detectable.
  * A precise rate limiter paces requests for the paid 75-req/min tier.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

AV_URL = "https://www.alphavantage.co/query"

# Project ticker -> Alpha Vantage query symbol (only where they differ).
AV_SYMBOL = {"GOOGL": "GOOG"}

# News-and-sentiment archive begins here; earlier requests return nothing.
ARCHIVE_START = "20220301T0000"

_REQUESTS_PER_MIN = 75
_MIN_INTERVAL = 60.0 / _REQUESTS_PER_MIN
_last_call = [0.0]


def _rate_limit() -> None:
    """Block so that requests are spaced for the configured per-minute tier."""
    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _parse_feed(feed: list[dict], project_ticker: str,
                api_symbol: str) -> list[dict]:
    """Convert one API `feed` array into rows in the project schema.

    Articles are matched on the AV query symbol, but the stored `ticker` is
    the project ticker so downstream joins to price data still line up.
    """
    rows = []
    for art in feed:
        av_score = av_rel = None
        for ts in art.get("ticker_sentiment", []):
            if ts.get("ticker") == api_symbol:
                av_score = float(ts.get("ticker_sentiment_score", "nan"))
                av_rel = float(ts.get("relevance_score", "nan"))
                break
        if av_score is None:            # ticker not actually tagged; skip
            continue
        rows.append({
            "published_at": pd.to_datetime(art.get("time_published"),
                                           format="%Y%m%dT%H%M%S",
                                           errors="coerce"),
            "ticker": project_ticker,
            "headline": art.get("title", ""),
            "summary": art.get("summary", ""),
            "source": art.get("source", "alphavantage"),
            "url": art.get("url", ""),
            "av_sentiment_score": av_score,
            "av_relevance": av_rel,
        })
    return rows


def fetch_ticker(project_ticker: str, time_from: str, time_to: str,
                 api_key: str, max_pages: int = 40) -> tuple[list[dict], bool]:
    """Paginate NEWS_SENTIMENT for one ticker over [time_from, time_to].

    Returns (rows, complete). `complete` is True only if pagination reached
    the end of the range; False means it stopped on max_pages, an empty or
    malformed cursor, or an API message -- i.e. a TRUNCATED pull to re-run.
    Datetime strings are 'YYYYMMDDTHHMM'.
    """
    api_symbol = AV_SYMBOL.get(project_ticker, project_ticker)
    rows: list[dict] = []
    cursor, complete = time_from, False

    for _ in range(max_pages):
        _rate_limit()
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": api_symbol,
            "time_from": cursor,
            "time_to": time_to,
            "limit": "1000",
            "sort": "EARLIEST",
            "apikey": api_key,
        }
        resp = requests.get(AV_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "feed" not in data:
            msg = (data.get("Note") or data.get("Information")
                   or data.get("Error Message") or str(data)[:150])
            logger.warning("[%s] API message: %s", project_ticker, msg)
            break

        feed = data["feed"]
        if not feed:
            complete = True
            break
        rows.extend(_parse_feed(feed, project_ticker, api_symbol))

        if len(feed) < 1000:
            complete = True
            break

        newest = max((a.get("time_published", "") for a in feed), default="")
        newest_ts = pd.to_datetime(newest, format="%Y%m%dT%H%M%S",
                                   errors="coerce")
        if pd.isna(newest_ts):
            logger.warning("[%s] unparseable cursor %r; stopping",
                           project_ticker, newest)
            break
        cursor = (newest_ts + pd.Timedelta(minutes=1)).strftime("%Y%m%dT%H%M")

    return rows, complete


def fetch_alphavantage(tickers: list[str], api_key: str,
                       start: str = ARCHIVE_START,
                       end: str = "20251231T0000",
                       max_pages: int = 40,
                       progress_dir=None,
                       refetch: list[str] | None = None) -> pd.DataFrame:
    """Fetch news for every ticker, caching per-ticker progress.

    Parameters
    ----------
    tickers : list[str]
        Project tickers to fetch.
    api_key : str
        Alpha Vantage API key.
    start, end : str
        'YYYYMMDDTHHMM' bounds. `start` defaults to the archive start.
    max_pages : int
        Page cap per ticker (1000 articles/page).
    progress_dir : Path | None
        If given, each ticker is cached to `<progress_dir>/<ticker>.parquet`
        and reused on re-runs; tickers in `refetch` are re-pulled.
    refetch : list[str] | None
        Tickers to force-refetch even if a progress file exists.

    Returns
    -------
    DataFrame in the project schema (deduplicated). A `_complete` attribute
    on the frame's .attrs records per-ticker completeness.
    """
    refetch = set(refetch or [])
    if progress_dir is not None:
        progress_dir.mkdir(parents=True, exist_ok=True)
        for t in refetch:
            stale = progress_dir / f"{t}.parquet"
            if stale.exists():
                stale.unlink()
                logger.info("cleared stale cache for %s", t)

    all_rows: list[dict] = []
    completeness: dict[str, tuple[str, int]] = {}

    for ticker in tickers:
        tcache = progress_dir / f"{ticker}.parquet" if progress_dir else None
        if tcache is not None and tcache.exists():
            df_t = pd.read_parquet(tcache)
            all_rows.extend(df_t.to_dict("records"))
            completeness[ticker] = ("cached", len(df_t))
            continue
        rows, complete = fetch_ticker(ticker, start, end, api_key,
                                      max_pages=max_pages)
        if rows and tcache is not None:
            pd.DataFrame(rows).to_parquet(tcache, index=False)
        all_rows.extend(rows)
        completeness[ticker] = ("complete" if complete else "TRUNCATED",
                                len(rows))
        logger.info("%s: %d articles [%s]", ticker, len(rows),
                    completeness[ticker][0])

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["ticker", "headline", "published_at"]).reset_index(drop=True)
    df.attrs["completeness"] = completeness
    return df


def completeness_report(df: pd.DataFrame, tickers: list[str],
                        end_floor: str = "2025-12-01") -> pd.DataFrame:
    """Per-ticker coverage table with heuristic flags for truncated pulls."""
    report = df.groupby("ticker").agg(
        n=("headline", "size"),
        earliest=("published_at", "min"),
        latest=("published_at", "max"),
    ).sort_values("n")
    report["reaches_end"] = report["latest"] >= pd.Timestamp(end_floor)
    report["round_1000"] = (report["n"] % 1000 == 0)
    report["suspect"] = (~report["reaches_end"]) | report["round_1000"]
    missing = [t for t in tickers if t not in report.index]
    report.attrs["missing"] = missing
    return report
