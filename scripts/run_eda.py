"""
Exploratory Data Analysis — produces the figures and summary statistics
for the Data Summary section of the report (Module 3 milestone).

Covers both data streams:
  Prices:   coverage matrix, return distributions, volatility regimes,
            cross-ticker return correlation heatmap.
  News:     article volume over time and per ticker, sentiment score
            distribution, sentiment vs. same-day / next-day returns.

Run after data is fetched:  python scripts/run_eda.py
(or on synthetic data by passing --synthetic)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FIGURES_DIR


def load(synthetic: bool):
    if synthetic:
        from scripts.demo_synthetic import make_synthetic_world
        prices, scored = make_synthetic_world()
    else:
        prices = pd.read_parquet(PROCESSED_DIR / "prices_clean.parquet")
        scored = pd.read_parquet(PROCESSED_DIR / "news_scored.parquet")
    return prices, scored


def eda_prices(prices: pd.DataFrame):
    px = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    rets = px.pct_change()

    # Return distribution + correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(rets.to_numpy().ravel(), bins=120, range=(-0.1, 0.1),
                 color="tab:blue", alpha=0.8)
    axes[0].set_title("Daily return distribution (all tickers)")
    axes[0].set_xlabel("Daily return")

    corr = rets.corr()
    im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    axes[1].set_yticks(range(len(corr)), corr.columns, fontsize=7)
    axes[1].set_title("Cross-ticker return correlation")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_prices.png", dpi=150)

    print("Price coverage:")
    print(prices.groupby("ticker")["date"]
          .agg(["min", "max", "count"]).to_string())


def eda_news(scored: pd.DataFrame, prices: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Article volume over time
    vol = scored.groupby(pd.Grouper(key="effective_date", freq="W")).size()
    axes[0].plot(vol.index, vol.values, color="tab:blue")
    axes[0].set_title("Weekly article volume")

    # Sentiment score distribution
    axes[1].hist(scored["score"], bins=60, color="tab:orange", alpha=0.85)
    axes[1].set_title("Article sentiment score distribution")
    axes[1].set_xlabel("P(pos) - P(neg)")

    # Sentiment vs next-day return (the motivating scatter)
    px = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    fwd = px.pct_change().shift(-1)  # next-day return
    daily = scored.groupby(["effective_date", "ticker"])["score"].mean()
    daily = daily.reset_index().rename(columns={"effective_date": "date"})
    daily["fwd_ret"] = [
        fwd.at[d, t] if (d in fwd.index and t in fwd.columns) else np.nan
        for d, t in zip(daily["date"], daily["ticker"])]
    daily = daily.dropna()
    axes[2].scatter(daily["score"], daily["fwd_ret"], s=4, alpha=0.25)
    rho = daily["score"].corr(daily["fwd_ret"], method="spearman")
    axes[2].set_title(f"Daily sentiment vs next-day return (ρ={rho:.3f})")
    axes[2].set_xlabel("Mean daily sentiment")
    axes[2].set_ylabel("Next-day return")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_news.png", dpi=150)
    print(f"\nArticles: {len(scored):,} | "
          f"tickers covered: {scored['ticker'].nunique()} | "
          f"Spearman(sentiment, next-day ret) = {rho:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    prices, scored = load(args.synthetic)
    eda_prices(prices)
    eda_news(scored, prices)
    print("EDA figures written to reports/figures/")
