"""
Signal generation: convert the daily sentiment panel + price features into
discrete trade entry signals.

Rule set (deliberately simple and interpretable — the experimental variable
in this project is the HOLDING HORIZON, not signal sophistication):

  LONG entry on day t when, using only information effective on or before t:
    * rolling sentiment  sent_roll  >  long_threshold
    * article support    n_roll     >= min_articles
    * trend filter       close > SMA(trend_filter_sma)   [optional]

  SHORT entry mirrors the above with sent_roll < short_threshold and
  close < SMA. Set SignalConfig.short_threshold = None for long-only.

Execution convention: signals observed at the close of day t are FILLED AT
THE NEXT DAY'S OPEN-proxy (we use next close in the simplified engine; see
backtest/engine.py for the exact fill model and its justification).
"""

from __future__ import annotations

import pandas as pd

from src.config import SIGNALS


def generate_signals(sent_panel: pd.DataFrame,
                     prices_feat: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame [date, ticker, direction, sent_roll, n_roll]
    with one row per entry signal (direction: +1 long, -1 short)."""
    df = sent_panel.merge(
        prices_feat[["date", "ticker", "adj_close",
                     f"sma_{SIGNALS.trend_filter_sma}", "above_sma20"]],
        on=["date", "ticker"], how="inner")

    long_ok = (
        (df["sent_roll"] > SIGNALS.long_threshold)
        & (df["n_roll"] >= SIGNALS.min_articles)
        & (df["above_sma20"] == 1)
    )
    signals = [df.loc[long_ok].assign(direction=1)]

    if SIGNALS.short_threshold is not None:
        short_ok = (
            (df["sent_roll"] < SIGNALS.short_threshold)
            & (df["n_roll"] >= SIGNALS.min_articles)
            & (df["above_sma20"] == 0)
        )
        signals.append(df.loc[short_ok].assign(direction=-1))

    out = pd.concat(signals, ignore_index=True)
    cols = ["date", "ticker", "direction", "sent_roll", "n_roll"]
    return out[cols].sort_values(["date", "ticker"]).reset_index(drop=True)
