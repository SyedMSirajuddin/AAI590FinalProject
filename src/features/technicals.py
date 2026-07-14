"""
Price-derived features used by (a) the trend filter in signal generation
and (b) the optional sequence model that combines sentiment with price
context. All features are computed from information available at or
before each row's date (shift-safe by construction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    """Append SMA, momentum, volatility, and RSI columns per ticker."""
    df = prices.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = g["adj_close"].pct_change()
    for w in (5, 20, 50):
        df[f"sma_{w}"] = g["adj_close"].apply(
            lambda s, w=w: s.rolling(w).mean())
    df["mom_21"] = g["adj_close"].apply(lambda s: s.pct_change(21))
    df["vol_21"] = g["ret_1d"].apply(
        lambda s: s.rolling(21).std() * np.sqrt(252))
    df["rsi_14"] = g["adj_close"].apply(_rsi)
    df["above_sma20"] = (df["adj_close"] > df["sma_20"]).astype(int)
    return df


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
