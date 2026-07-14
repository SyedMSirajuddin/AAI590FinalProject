"""
Price data ingestion and cleaning.

Downloads daily OHLCV bars for the configured universe via yfinance,
performs data-quality checks, and writes a clean long-format parquet file:
    columns = [date, ticker, open, high, low, close, adj_close, volume]

Cleaning steps (documented for the Data Summary section of the report):
  1. Drop rows with any missing OHLC value (rare; typically halted sessions).
  2. Forward-fill volume gaps of a single day; drop longer gaps.
  3. Remove tickers with < min_history trading days (insufficient for
     the longest holding horizon under study).
  4. Flag and winsorize extreme single-day returns (> |50%|) that are not
     corroborated by volume spikes — these are almost always bad prints
     or unadjusted split artifacts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR, UNIVERSE

logger = logging.getLogger(__name__)

MIN_HISTORY = 252          # at least one year of data
EXTREME_RETURN = 0.50      # |daily return| beyond this is suspect
VOLUME_SPIKE_MULT = 3.0    # a real event usually carries >3x median volume


def download_prices(tickers: list[str] | None = None,
                    start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV bars with yfinance (auto-adjusted=False so we
    keep both raw close and adjusted close)."""
    import yfinance as yf  # imported lazily; not available in all sandboxes

    tickers = tickers or UNIVERSE.tickers
    start = start or UNIVERSE.start_date
    end = end or UNIVERSE.end_date

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=False, group_by="ticker", progress=False)

    frames = []
    for t in tickers:
        df = raw[t].copy() if len(tickers) > 1 else raw.copy()
        df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        df["ticker"] = t
        frames.append(df.reset_index().rename(columns={"Date": "date"}))

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out.to_parquet(RAW_DIR / "prices_raw.parquet", index=False)
    logger.info("Downloaded %d rows for %d tickers", len(out), len(tickers))
    return out


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning steps described in the module docstring."""
    df = df.copy().sort_values(["ticker", "date"])

    # 1. Missing OHLC
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    logger.info("Dropped %d rows with missing OHLC", before - len(df))

    # 2. Volume gaps
    df["volume"] = df.groupby("ticker")["volume"].transform(
        lambda s: s.ffill(limit=1))
    df = df.dropna(subset=["volume"])

    # 3. Minimum history
    counts = df.groupby("ticker")["date"].transform("count")
    df = df[counts >= MIN_HISTORY]

    # 4. Extreme returns not backed by volume
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()
    med_vol = df.groupby("ticker")["volume"].transform(
        lambda s: s.rolling(63, min_periods=20).median())
    suspicious = (df["ret_1d"].abs() > EXTREME_RETURN) & \
                 (df["volume"] < VOLUME_SPIKE_MULT * med_vol)
    n_susp = int(suspicious.sum())
    if n_susp:
        logger.warning("Winsorizing %d suspicious extreme-return rows", n_susp)
        df.loc[suspicious, "ret_1d"] = np.clip(
            df.loc[suspicious, "ret_1d"], -EXTREME_RETURN, EXTREME_RETURN)

    df.to_parquet(PROCESSED_DIR / "prices_clean.parquet", index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices = download_prices()
    clean_prices(prices)
