"""
Synthetic-data demo: exercises the ENTIRE downstream pipeline
(aggregation -> signals -> multi-horizon backtest -> null baseline ->
decay analysis -> figures) without network access or GPU.

The synthetic world is built with a KNOWN, PLANTED effect so the demo
doubles as a validation test of the whole apparatus:
  * Prices follow geometric random walks.
  * "News sentiment" is generated with genuine but DECAYING predictive
    power: a positive-sentiment day nudges the next ~5 days of returns
    upward, with the effect shrinking geometrically.
If the pipeline is correct, the recovered decay curve should show strong
short-horizon performance fading toward the null band at long horizons —
i.e., the machinery recovers the effect we planted. This is exactly the
validation argument to make in the Methodology section.

Run:  python scripts/demo_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import BACKTEST, UNIVERSE, PROCESSED_DIR
from src.features.aggregate import daily_sentiment_panel
from src.features.technicals import add_technicals
from src.signals.generate import generate_signals
from src.backtest.engine import run_all_horizons
from src.backtest.baselines import buy_and_hold, random_signals_null
from src.analysis.horizon_decay import (horizon_table, event_study,
                                        plot_decay_curve, plot_event_study,
                                        plot_equity_curves)

RNG = np.random.default_rng(7)
N_DAYS = 1000
EFFECT_DAYS = 5          # planted signal persists ~5 trading days
EFFECT_SIZE = 0.008      # +80 bps expected drift on day 1, decaying 40%/day
NEWS_PROB = 0.15         # chance a ticker has a news day


def make_synthetic_world():
    dates = pd.bdate_range("2021-01-04", periods=N_DAYS)
    tickers = UNIVERSE.tickers

    price_rows, news_rows = [], []
    for t in tickers:
        # planted sentiment impulses
        has_news = RNG.random(N_DAYS) < NEWS_PROB
        polarity = RNG.choice([1.0, -1.0], size=N_DAYS)
        impulse = np.where(has_news, polarity, 0.0)

        # decaying drift injected into returns
        drift = np.zeros(N_DAYS)
        for lag in range(1, EFFECT_DAYS + 1):
            drift[lag:] += impulse[:-lag] * EFFECT_SIZE * (0.6 ** (lag - 1))

        rets = RNG.normal(0.0004, 0.02, N_DAYS) + drift
        close = 100 * np.cumprod(1 + rets)
        price_rows.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adj_close": close,
            "volume": RNG.integers(1e6, 5e6, N_DAYS).astype(float),
        }))

        # articles consistent with the impulse (plus label noise)
        for i in np.flatnonzero(has_news):
            n_arts = RNG.integers(2, 6)
            for _ in range(n_arts):
                true_pos = impulse[i] > 0
                noisy_pos = true_pos if RNG.random() > 0.1 else not true_pos
                p_pos = RNG.uniform(0.55, 0.95) if noisy_pos \
                    else RNG.uniform(0.02, 0.25)
                p_neg = (1 - p_pos) * RNG.uniform(0.5, 0.9)
                news_rows.append({
                    "effective_date": dates[i], "ticker": t,
                    "headline": f"synthetic article {t} {dates[i].date()}",
                    "p_positive": p_pos, "p_negative": p_neg,
                    "p_neutral": 1 - p_pos - p_neg,
                    "score": p_pos - p_neg,
                })

    return pd.concat(price_rows, ignore_index=True), pd.DataFrame(news_rows)


def main():
    prices, scored = make_synthetic_world()
    trading_days = pd.DatetimeIndex(prices["date"].drop_duplicates())

    panel = daily_sentiment_panel(scored, trading_days, UNIVERSE.tickers)
    prices_feat = add_technicals(prices)
    signals = generate_signals(panel, prices_feat)
    print(f"Signals generated: {len(signals)} "
          f"({(signals['direction'] == 1).sum()} long / "
          f"{(signals['direction'] == -1).sum()} short)")

    results = run_all_horizons(signals, prices)
    nulls = {h: random_signals_null(signals, prices, h, n_sims=30)
             for h in BACKTEST.horizons}

    table = horizon_table(results, nulls)
    table.to_csv(PROCESSED_DIR / "horizon_table_synthetic.csv")
    print("\n=== Horizon comparison (synthetic world, planted 5-day effect) ===")
    cols = ["strategy", "n_trades", "hit_rate", "avg_trade_ret",
            "sharpe", "null_sharpe_p95", "beats_null_p95", "max_drawdown"]
    print(table[[c for c in cols if c in table.columns]].round(3).to_string())

    plot_decay_curve(table, fname="decay_sharpe_synthetic.png")
    plot_event_study(event_study(signals, prices, max_days=126),
                     fname="event_study_synthetic.png")
    plot_equity_curves(results, buy_and_hold(prices),
                       fname="equity_curves_synthetic.png")
    print("\nFigures written to reports/figures/*_synthetic.png")


if __name__ == "__main__":
    main()
