"""
Aggregate article-level sentiment to a daily, per-ticker panel.

Output schema (one row per ticker × trading day):
    n_articles          — article count on the effective date
    sent_mean           — mean scalar score of the day's articles
    sent_weighted       — confidence-weighted mean (weight = 1 - p_neutral,
                          so decisive articles count more than ambiguous ones)
    sent_roll           — rolling mean of sent_weighted over the signal window
    n_roll              — rolling article count over the same window

The rolling features use only current and PAST effective dates, preserving
the point-in-time guarantee established at ingestion.
"""

from __future__ import annotations

import pandas as pd

from src.config import SIGNALS, PROCESSED_DIR


def daily_sentiment_panel(scored_news: pd.DataFrame,
                          trading_days: pd.DatetimeIndex,
                          tickers: list[str]) -> pd.DataFrame:
    """Collapse scored articles into a dense daily panel (missing days = 0
    articles, NaN sentiment) and add rolling-window features."""
    g = scored_news.groupby(["ticker", "effective_date"])
    daily = g.agg(
        n_articles=("score", "size"),
        sent_mean=("score", "mean"),
    ).reset_index()

    # Confidence-weighted mean: sum(score * conf) / sum(conf)
    scored = scored_news.assign(conf=1.0 - scored_news["p_neutral"],
                                wscore=lambda d: d["score"] * d["conf"])
    w = scored.groupby(["ticker", "effective_date"]).agg(
        wsum=("wscore", "sum"), csum=("conf", "sum")).reset_index()
    w["sent_weighted"] = w["wsum"] / w["csum"].replace(0, pd.NA)
    daily = daily.merge(w[["ticker", "effective_date", "sent_weighted"]],
                        on=["ticker", "effective_date"], how="left")

    # Dense grid so rolling windows are calendar-consistent
    grid = pd.MultiIndex.from_product(
        [tickers, trading_days], names=["ticker", "date"]).to_frame(index=False)
    panel = grid.merge(
        daily.rename(columns={"effective_date": "date"}),
        on=["ticker", "date"], how="left")
    panel["n_articles"] = panel["n_articles"].fillna(0).astype(int)

    win = SIGNALS.sentiment_window
    grp = panel.groupby("ticker", group_keys=False)
    panel["sent_roll"] = grp["sent_weighted"].apply(
        lambda s: s.rolling(win, min_periods=1).mean())
    panel["n_roll"] = grp["n_articles"].apply(
        lambda s: s.rolling(win, min_periods=1).sum())

    panel.to_parquet(PROCESSED_DIR / "sentiment_panel.parquet", index=False)
    return panel
