"""
Point-in-time, fixed-horizon backtest engine.

This is the experimental core of the project: the SAME signal stream is
evaluated at multiple fixed holding horizons, isolating the horizon as the
only variable. For each horizon H:

  * A signal on day t is filled at the close of day t+1 (one-bar delay —
    a conservative proxy that guarantees the information was available
    before execution and removes same-close look-ahead).
  * The position is exited at the close of day t+1+H.
  * Per-trade return = direction * (exit/entry - 1) - 2 * cost_per_side.
  * Portfolio return series: each day's return is the equal-weighted mean
    of the daily returns of all open positions (capped at max_positions,
    first-come priority), with idle capital earning 0.

Two result granularities are produced:
  trades_df     — one row per trade (for hit-rate / distribution analysis)
  daily_returns — portfolio daily return series (for Sharpe / drawdown)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import BACKTEST


@dataclass
class HorizonResult:
    horizon: int
    trades: pd.DataFrame
    daily_returns: pd.Series


def _price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot to a date × ticker matrix of adjusted closes."""
    return prices.pivot_table(index="date", columns="ticker",
                              values="adj_close").sort_index()


def run_horizon(signals: pd.DataFrame, prices: pd.DataFrame,
                horizon: int,
                cost_per_side: float | None = None,
                max_positions: int | None = None) -> HorizonResult:
    """Backtest one fixed holding horizon. See module docstring for rules."""
    cost = BACKTEST.cost_per_side if cost_per_side is None else cost_per_side
    max_pos = max_positions or BACKTEST.max_positions

    px = _price_matrix(prices)
    dates = px.index
    date_pos = pd.Series(np.arange(len(dates)), index=dates)

    trades = []
    open_until = {}  # ticker -> exit date index, to prevent overlapping entries
    n_open_by_day = np.zeros(len(dates), dtype=int)

    for row in signals.itertuples(index=False):
        if row.date not in date_pos.index or row.ticker not in px.columns:
            continue
        t = int(date_pos[row.date])
        entry_i, exit_i = t + 1, t + 1 + horizon
        if exit_i >= len(dates):
            continue  # trade would extend past available history

        # No pyramiding: skip if this ticker already has an open position
        if open_until.get(row.ticker, -1) >= entry_i:
            continue
        # Portfolio capacity: cap simultaneous open positions
        if n_open_by_day[entry_i:exit_i].max(initial=0) >= max_pos:
            continue

        entry_px = px.iloc[entry_i][row.ticker]
        exit_px = px.iloc[exit_i][row.ticker]
        if np.isnan(entry_px) or np.isnan(exit_px):
            continue

        gross = row.direction * (exit_px / entry_px - 1.0)
        net = gross - 2.0 * cost
        trades.append({
            "ticker": row.ticker, "direction": row.direction,
            "signal_date": row.date,
            "entry_date": dates[entry_i], "exit_date": dates[exit_i],
            "entry_px": entry_px, "exit_px": exit_px,
            "gross_ret": gross, "net_ret": net,
        })
        open_until[row.ticker] = exit_i
        n_open_by_day[entry_i:exit_i] += 1

    trades_df = pd.DataFrame(trades)

    # ---- daily portfolio return series ------------------------------------
    daily = pd.Series(0.0, index=dates)
    if not trades_df.empty:
        ret_1d = px.pct_change()
        weight = pd.DataFrame(0.0, index=dates, columns=px.columns)
        for tr in trades_df.itertuples(index=False):
            ei, xi = int(date_pos[tr.entry_date]), int(date_pos[tr.exit_date])
            # returns accrue from the bar AFTER entry through the exit bar
            weight.iloc[ei + 1: xi + 1,
                        weight.columns.get_loc(tr.ticker)] += tr.direction
        n_open = weight.abs().sum(axis=1).replace(0, np.nan)
        daily = (weight * ret_1d).sum(axis=1) / n_open
        daily = daily.fillna(0.0)
        # Spread each trade's round-trip cost over its entry day
        cost_hits = trades_df.groupby("entry_date").size() * 2 * cost
        per_day_open = n_open.reindex(cost_hits.index).fillna(1)
        daily.loc[cost_hits.index] -= (cost_hits / per_day_open).values

    return HorizonResult(horizon=horizon, trades=trades_df, daily_returns=daily)


def run_all_horizons(signals: pd.DataFrame, prices: pd.DataFrame,
                     horizons: list[int] | None = None) -> dict[int, HorizonResult]:
    """Run the engine across the full horizon grid from config."""
    horizons = horizons or BACKTEST.horizons
    return {h: run_horizon(signals, prices, h) for h in horizons}
