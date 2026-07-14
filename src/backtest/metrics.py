"""
Performance metrics reported for every strategy/horizon:
cumulative return, annualized return & volatility, Sharpe ratio,
maximum drawdown, hit rate, and average trade return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BACKTEST


def cumulative_return(daily: pd.Series) -> float:
    return float((1.0 + daily).prod() - 1.0)


def annualized_return(daily: pd.Series) -> float:
    n = len(daily)
    if n == 0:
        return np.nan
    total = (1.0 + daily).prod()
    return float(total ** (BACKTEST.trading_days / n) - 1.0)


def annualized_vol(daily: pd.Series) -> float:
    return float(daily.std() * np.sqrt(BACKTEST.trading_days))


def sharpe_ratio(daily: pd.Series) -> float:
    vol = daily.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    excess = daily.mean() - BACKTEST.risk_free_rate / BACKTEST.trading_days
    return float(np.sqrt(BACKTEST.trading_days) * excess / vol)


def max_drawdown(daily: pd.Series) -> float:
    """Most negative peak-to-trough decline of the equity curve."""
    equity = (1.0 + daily).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def summarize(daily: pd.Series, trades: pd.DataFrame | None = None,
              label: str = "") -> dict:
    """One row of the horizon-comparison table."""
    out = {
        "strategy": label,
        "cum_return": cumulative_return(daily),
        "ann_return": annualized_return(daily),
        "ann_vol": annualized_vol(daily),
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": max_drawdown(daily),
    }
    if trades is not None and not trades.empty:
        out["n_trades"] = len(trades)
        out["hit_rate"] = float((trades["net_ret"] > 0).mean())
        out["avg_trade_ret"] = float(trades["net_ret"].mean())
    return out
