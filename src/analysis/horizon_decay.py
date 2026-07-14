"""
Horizon-decay analysis: the project's headline result.

Builds the comparison table and the decay-curve figures that directly test
the hypothesis — that sentiment signal strength is greatest at short
horizons and decays as the holding period lengthens.

Three complementary views:
  1. Strategy metrics vs. horizon (Sharpe, avg trade return, hit rate),
     with the random-signal null band overlaid.
  2. Event-study curve: mean cumulative return following a signal, day by
     day out to the longest horizon (the "shape" of the decay).
  3. Equity curves for selected horizons vs. buy-and-hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import HorizonResult
from src.backtest.metrics import summarize
from src.config import FIGURES_DIR


def horizon_table(results: dict[int, HorizonResult],
                  nulls: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Metrics-per-horizon table; adds null-distribution percentiles if given."""
    rows = []
    for h, res in sorted(results.items()):
        row = summarize(res.daily_returns, res.trades, label=f"H={h}d")
        row["horizon"] = h
        if nulls and h in nulls and not nulls[h].empty:
            null_sharpes = nulls[h]["sharpe"]
            row["null_sharpe_median"] = float(null_sharpes.median())
            row["null_sharpe_p95"] = float(null_sharpes.quantile(0.95))
            row["beats_null_p95"] = row["sharpe"] > row["null_sharpe_p95"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")


def event_study(signals: pd.DataFrame, prices: pd.DataFrame,
                max_days: int = 126) -> pd.DataFrame:
    """Mean (and IQR) cumulative signed return path following each signal.

    This is the cleanest visualization of predictive decay: if sentiment
    carries short-lived information, the mean path rises early then
    flattens (or mean-reverts) as the horizon extends.
    """
    px = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    dates = px.index
    date_pos = pd.Series(np.arange(len(dates)), index=dates)

    paths = []
    for row in signals.itertuples(index=False):
        if row.date not in date_pos.index or row.ticker not in px.columns:
            continue
        t = int(date_pos[row.date])
        entry_i = t + 1
        if entry_i + max_days >= len(dates):
            continue
        window = px[row.ticker].iloc[entry_i: entry_i + max_days + 1].to_numpy()
        if np.isnan(window).any() or window[0] == 0:
            continue
        paths.append(row.direction * (window / window[0] - 1.0))

    arr = np.vstack(paths) if paths else np.empty((0, max_days + 1))
    days = np.arange(max_days + 1)
    return pd.DataFrame({
        "day": days,
        "mean": arr.mean(axis=0) if len(arr) else np.nan,
        "q25": np.quantile(arr, 0.25, axis=0) if len(arr) else np.nan,
        "q75": np.quantile(arr, 0.75, axis=0) if len(arr) else np.nan,
        "n": len(arr),
    })


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_decay_curve(table: pd.DataFrame, fname: str = "decay_sharpe.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [("sharpe", "Sharpe ratio"),
               ("avg_trade_ret", "Avg net return per trade"),
               ("hit_rate", "Hit rate")]
    for ax, (col, title) in zip(axes, metrics):
        if col in table:
            ax.plot(table.index, table[col], marker="o", color="tab:blue",
                    label="sentiment strategy")
        if col == "sharpe" and "null_sharpe_p95" in table:
            ax.fill_between(table.index, table["null_sharpe_median"],
                            table["null_sharpe_p95"], alpha=0.25,
                            color="grey", label="random-signal null (50–95th pct)")
            ax.legend(fontsize=8)
        if col == "hit_rate":
            ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(table.index)
        ax.set_xticklabels(table.index)
        ax.set_xlabel("Holding horizon (trading days)")
        ax.set_title(title)
    fig.suptitle("Predictive decay of news sentiment across holding horizons")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    return fig


def plot_event_study(es: pd.DataFrame, fname: str = "event_study.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(es["day"], es["mean"] * 100, color="tab:blue", label="mean path")
    ax.fill_between(es["day"], es["q25"] * 100, es["q75"] * 100,
                    alpha=0.2, color="tab:blue", label="IQR")
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_xlabel("Trading days after signal")
    ax.set_ylabel("Cumulative signed return (%)")
    ax.set_title(f"Post-signal return path (n={int(es['n'].iloc[0])} signals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    return fig


def plot_equity_curves(results, bh: pd.Series,
                       horizons_to_show=(5, 21, 63),
                       fname: str = "equity_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for h in horizons_to_show:
        if h in results:
            eq = (1 + results[h].daily_returns).cumprod()
            ax.plot(eq.index, eq, label=f"sentiment H={h}d")
    ax.plot(bh.index, (1 + bh).cumprod(), color="black", ls="--",
            label="buy & hold (equal-weight)")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Equity curves: sentiment strategy vs. buy-and-hold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    return fig
