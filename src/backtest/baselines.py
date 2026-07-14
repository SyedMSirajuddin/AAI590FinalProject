"""
Baseline strategies the sentiment strategy must beat (or fail to beat —
either outcome is informative under the project's hypothesis):

  1. buy_and_hold      — hold the benchmark (SPY) or the equal-weighted
                         universe for the whole test period.
  2. random_signals    — the "no-skill" benchmark: same NUMBER of trades,
                         same horizon, same cost model, but entry dates and
                         tickers drawn uniformly at random. Repeated N times
                         to produce a null distribution — the sentiment
                         strategy's Sharpe is then judged against this
                         distribution rather than against zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import run_horizon, HorizonResult


def buy_and_hold(prices: pd.DataFrame,
                 tickers: list[str] | None = None) -> pd.Series:
    """Equal-weight daily return series for the given tickers (or all)."""
    px = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    if tickers:
        px = px[[t for t in tickers if t in px.columns]]
    return px.pct_change().mean(axis=1).fillna(0.0)


def random_signals_null(signals: pd.DataFrame, prices: pd.DataFrame,
                        horizon: int, n_sims: int = 100,
                        seed: int = 42) -> pd.DataFrame:
    """Monte-Carlo null: shuffle signal dates/tickers, keep count constant.

    Returns a DataFrame with one row per simulation and summary columns
    (mean net return per trade, total return, Sharpe proxy).
    """
    rng = np.random.default_rng(seed)
    dates = prices["date"].drop_duplicates().sort_values().to_numpy()
    tickers = prices["ticker"].unique()
    n = len(signals)

    rows = []
    for i in range(n_sims):
        fake = pd.DataFrame({
            "date": rng.choice(dates[:-horizon - 2], size=n),
            "ticker": rng.choice(tickers, size=n),
            "direction": rng.choice([1, -1], size=n)
            if (signals["direction"] == -1).any()
            else np.ones(n, dtype=int),
        }).sort_values("date")
        res: HorizonResult = run_horizon(fake, prices, horizon)
        r = res.daily_returns
        rows.append({
            "sim": i,
            "n_trades": len(res.trades),
            "mean_trade_ret": res.trades["net_ret"].mean()
            if not res.trades.empty else np.nan,
            "total_return": (1 + r).prod() - 1,
            "sharpe": np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0.0,
        })
    return pd.DataFrame(rows)
