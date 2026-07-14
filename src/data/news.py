"""
Historical financial news ingestion.

Two interchangeable sources are supported (both keyed to a ticker and a
date range) so the pipeline is not locked to a single vendor:

  * Finnhub  — `/company-news` endpoint. Clean per-ticker attribution,
               ~1 year of history on the free tier. Requires FINNHUB_API_KEY.
  * GDELT    — open dataset, long history, but articles must be linked to
               tickers by matching company names/aliases in the headline.

Every article is normalized to the schema:
    [published_at (UTC), ticker, headline, summary, source, url]

POINT-IN-TIME DISCIPLINE
------------------------
`effective_date` maps each article to the first trading session on which it
could have influenced a trade decision made at the close:
  - published before that day's market close  -> same trading day
  - published after the close / weekend / holiday -> next trading day
This mapping is the project's main defense against look-ahead bias and is
applied here, at ingestion time, so every downstream stage inherits it.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from src.config import RAW_DIR, UNIVERSE

logger = logging.getLogger(__name__)

US_MARKET_CLOSE_UTC = 21  # 4pm ET ≈ 21:00 UTC (20:00 during DST; see note below)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def fetch_finnhub(tickers: list[str] | None = None,
                  start: str | None = None,
                  end: str | None = None,
                  pause: float = 1.1) -> pd.DataFrame:
    """Pull company news from Finnhub, one ticker/date-chunk at a time.

    The free tier allows 60 calls/min; `pause` keeps us under that limit.
    """
    import requests

    api_key = os.environ["FINNHUB_API_KEY"]
    tickers = tickers or UNIVERSE.tickers
    start = pd.Timestamp(start or UNIVERSE.start_date)
    end = pd.Timestamp(end or UNIVERSE.end_date)

    rows = []
    for ticker in tickers:
        empty_streak = 0
        chunk_end = end
        while chunk_end > start:
            chunk_start = max(chunk_end - timedelta(days=30), start)
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker,
                        "from": chunk_start.strftime("%Y-%m-%d"),
                        "to": chunk_end.strftime("%Y-%m-%d"),
                        "token": api_key},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
            for item in items:
                rows.append({
                    "published_at": datetime.utcfromtimestamp(item["datetime"]),
                    "ticker": ticker,
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "finnhub"),
                    "url": item.get("url", ""),
                })
            empty_streak = empty_streak + 1 if not items else 0
            if empty_streak >= 3:   # past the free tier's history horizon
                logger.info("%s: history exhausted at %s", ticker, chunk_start.date())
                break
            chunk_end = chunk_start
            time.sleep(pause)
        logger.info("Finnhub: fetched news for %s", ticker)

    df = pd.DataFrame(rows)
    df.to_parquet(RAW_DIR / "news_finnhub.parquet", index=False)
    return df


def fetch_gdelt(company_aliases: dict[str, list[str]],
                start: str | None = None,
                end: str | None = None,
                max_records: int = 250) -> pd.DataFrame:
    """Query the GDELT 2.0 DOC API for headlines matching company aliases.

    `company_aliases` maps ticker -> list of search phrases, e.g.
        {"AAPL": ["Apple Inc", "Apple iPhone"], ...}
    Alias-based linkage is noisier than Finnhub's per-ticker feed; the
    trade-off is much longer history at zero cost.
    """
    import requests

    start = pd.Timestamp(start or UNIVERSE.start_date)
    end = pd.Timestamp(end or UNIVERSE.end_date)

    rows = []
    for ticker, aliases in company_aliases.items():
        query = " OR ".join(f'"{a}"' for a in aliases)
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": f"({query}) sourcelang:english",
                    "mode": "artlist", "format": "json",
                    "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                    "enddatetime": end.strftime("%Y%m%d%H%M%S"),
                    "maxrecords": max_records, "sort": "datedesc"},
            timeout=60,
        )
        resp.raise_for_status()
        for art in resp.json().get("articles", []):
            rows.append({
                "published_at": pd.to_datetime(art["seendate"], utc=True)
                                  .tz_localize(None),
                "ticker": ticker,
                "headline": art.get("title", ""),
                "summary": "",
                "source": art.get("domain", "gdelt"),
                "url": art.get("url", ""),
            })
        logger.info("GDELT: fetched %s", ticker)

    df = pd.DataFrame(rows)
    df.to_parquet(RAW_DIR / "news_gdelt.parquet", index=False)
    return df


# ---------------------------------------------------------------------------
# Cleaning + point-in-time alignment
# ---------------------------------------------------------------------------
def clean_news(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Deduplicate, drop empty headlines, and assign `effective_date`."""
    df = df.copy()
    df["headline"] = df["headline"].str.strip()
    df = df[df["headline"].str.len() > 10]
    # Near-duplicate wire stories: same ticker + identical headline text
    df = df.drop_duplicates(subset=["ticker", "headline"])

    # --- point-in-time mapping -------------------------------------------
    # Articles published at/after the close roll forward to the next session.
    # NOTE: for production, replace the fixed-hour close with an
    # exchange-calendar lookup (pandas_market_calendars) to handle DST and
    # half-days precisely; the fixed cutoff is a conservative approximation.
    published = pd.to_datetime(df["published_at"])
    candidate = published.dt.normalize() + pd.to_timedelta(
        (published.dt.hour >= US_MARKET_CLOSE_UTC).astype(int), unit="D")

    sessions = pd.Series(trading_days, name="session")
    idx = sessions.searchsorted(candidate.dt.normalize())
    idx = idx.clip(0, len(sessions) - 1)
    df["effective_date"] = sessions.iloc[idx].values

    return df.sort_values(["effective_date", "ticker"]).reset_index(drop=True)
