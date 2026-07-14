"""
End-to-end pipeline runner (live-data mode).

Stages (each stage caches to data/processed so reruns are incremental):
  1. Download + clean prices           (src.data.prices)
  2. Fetch + clean news                (src.data.news)      [needs API key]
  3. Score news with FinBERT           (src.sentiment.finbert)
  4. Aggregate daily sentiment panel   (src.features.aggregate)
  5. Compute technicals + signals      (src.features / src.signals)
  6. Backtest across horizons + nulls  (src.backtest)
  7. Produce tables + figures          (src.analysis.horizon_decay)

Usage:
  export FINNHUB_API_KEY=...
  python scripts/run_pipeline.py                  # full run
  python scripts/run_pipeline.py --skip-fetch     # reuse cached data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, UNIVERSE, BACKTEST
from src.features.aggregate import daily_sentiment_panel
from src.features.technicals import add_technicals
from src.signals.generate import generate_signals
from src.backtest.engine import run_all_horizons
from src.backtest.baselines import buy_and_hold, random_signals_null
from src.analysis.horizon_decay import (horizon_table, event_study,
                                        plot_decay_curve, plot_event_study,
                                        plot_equity_curves)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")


def main(skip_fetch: bool = False, n_null_sims: int = 100):
    # ---- 1–2. Data --------------------------------------------------------
    if skip_fetch and (PROCESSED_DIR / "prices_clean.parquet").exists():
        prices = pd.read_parquet(PROCESSED_DIR / "prices_clean.parquet")
        news = pd.read_parquet(RAW_DIR / "news_finnhub.parquet")
    else:
        from src.data.prices import download_prices, clean_prices
        from src.data.news import fetch_finnhub
        prices = clean_prices(download_prices())
        news = fetch_finnhub()

    trading_days = pd.DatetimeIndex(
        prices["date"].drop_duplicates().sort_values())

    # ---- 3. Sentiment scoring --------------------------------------------
    scored_path = PROCESSED_DIR / "news_scored.parquet"
    if skip_fetch and scored_path.exists():
        scored = pd.read_parquet(scored_path)
    else:
        from src.data.news import clean_news
        from src.sentiment.finbert import FinBertScorer
        news = clean_news(news, trading_days)
        scored = FinBertScorer().score_news(news)
        scored.to_parquet(scored_path, index=False)

    # ---- 4–5. Features + signals -----------------------------------------
    panel = daily_sentiment_panel(scored, trading_days, UNIVERSE.tickers)
    prices_feat = add_technicals(prices)
    signals = generate_signals(panel, prices_feat)
    logger.info("Generated %d signals", len(signals))

    # Out-of-sample discipline: tune thresholds on pre-oos data only,
    # report headline results on the OOS window.
    oos = pd.Timestamp(BACKTEST.oos_start)
    signals_oos = signals[signals["date"] >= oos]
    logger.info("Of which %d are out-of-sample (>= %s)",
                len(signals_oos), BACKTEST.oos_start)

    # ---- 6. Backtests ------------------------------------------------------
    results = run_all_horizons(signals_oos, prices)
    nulls = {h: random_signals_null(signals_oos, prices, h,
                                    n_sims=n_null_sims)
             for h in BACKTEST.horizons}

    # ---- 7. Reporting ------------------------------------------------------
    table = horizon_table(results, nulls)
    table.to_csv(PROCESSED_DIR / "horizon_table.csv")
    print("\n=== Horizon comparison (out-of-sample) ===")
    print(table.round(3).to_string())

    plot_decay_curve(table)
    plot_event_study(event_study(signals_oos, prices,
                                 max_days=max(BACKTEST.horizons)))
    plot_equity_curves(results, buy_and_hold(prices))
    logger.info("Figures written to reports/figures/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--null-sims", type=int, default=100)
    a = p.parse_args()
    main(skip_fetch=a.skip_fetch, n_null_sims=a.null_sims)
