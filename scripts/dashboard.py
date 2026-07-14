"""
Decision-support dashboard (Streamlit) — the end-user deliverable from the
proposal: for a chosen date, show candidate trades flagged by the sentiment
screener, the horizon each signal historically supports best, and overall
backtest performance per horizon.

Run:  streamlit run scripts/dashboard.py
Requires the pipeline (or the synthetic demo) to have been run first so
that data/processed/ contains the panel, signals, and horizon table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR, FIGURES_DIR

st.set_page_config(page_title="Sentiment Horizon Screener", layout="wide")
st.title("News Sentiment Trading Screener")
st.caption("Research/decision-support tool — not investment advice.")


@st.cache_data
def load_artifacts():
    panel = pd.read_parquet(PROCESSED_DIR / "sentiment_panel.parquet")
    table_path = PROCESSED_DIR / "horizon_table.csv"
    if not table_path.exists():
        table_path = PROCESSED_DIR / "horizon_table_synthetic.csv"
    table = pd.read_csv(table_path, index_col=0)
    return panel, table


panel, table = load_artifacts()

# ---- Sidebar: date picker ---------------------------------------------------
dates = sorted(panel["date"].unique())
sel_date = st.sidebar.selectbox("Screening date", dates, index=len(dates) - 1)
min_sent = st.sidebar.slider("Min |rolling sentiment|", 0.0, 1.0, 0.35, 0.05)
min_arts = st.sidebar.slider("Min articles in window", 1, 10, 2)

# ---- Candidate trades -------------------------------------------------------
st.subheader(f"Candidates for {pd.Timestamp(sel_date).date()}")
day = panel[panel["date"] == sel_date].copy()
day = day[(day["sent_roll"].abs() >= min_sent) & (day["n_roll"] >= min_arts)]
day["direction"] = day["sent_roll"].apply(lambda s: "LONG" if s > 0 else "SHORT")

if day.empty:
    st.info("No tickers pass the sentiment screen on this date.")
else:
    best_h = table["sharpe"].idxmax() if "sharpe" in table else None
    st.dataframe(
        day[["ticker", "direction", "sent_roll", "n_roll"]]
        .sort_values("sent_roll", key=abs, ascending=False)
        .rename(columns={"sent_roll": "rolling sentiment",
                         "n_roll": "articles (window)"}),
        use_container_width=True)
    if best_h is not None:
        st.success(f"Historically best-supported holding horizon "
                   f"(by out-of-sample Sharpe): **{best_h} trading days**")

# ---- Horizon performance ----------------------------------------------------
st.subheader("Backtested performance by holding horizon")
st.dataframe(table.round(3), use_container_width=True)

for img in ("decay_sharpe.png", "decay_sharpe_synthetic.png"):
    if (FIGURES_DIR / img).exists():
        st.image(str(FIGURES_DIR / img))
        break
