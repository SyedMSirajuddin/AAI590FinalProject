"""
Optional second deep-learning model: an LSTM that combines rolling
sentiment features with price/technical features to predict the sign of
the forward H-day return. This gives the project a model-comparison axis
(rule-based signals vs. learned signals) on top of the horizon axis.

Design (report-ready description):
  Input:   sequences of L=20 trading days × F features per ticker
           (sent_roll, n_roll, ret_1d, mom_21, vol_21, rsi_14/100)
  Model:   LSTM(64) -> LSTM(32) -> Dense(16, ReLU) -> Dense(1, sigmoid)
  Target:  1 if forward H-day return > 0 else 0
  Loss:    binary cross-entropy; Adam(1e-3); early stopping on val AUC.
  Split:   STRICTLY chronological (train < oos_start <= test) — random
           splits leak future information in financial time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BACKTEST

SEQ_LEN = 20
FEATURES = ["sent_roll", "n_roll", "ret_1d", "mom_21", "vol_21", "rsi_14"]


def build_dataset(panel: pd.DataFrame, prices_feat: pd.DataFrame,
                  horizon: int = 5):
    """Build (X, y, meta) arrays of shape (N, SEQ_LEN, F) / (N,) / (N, 2).

    Normalization uses statistics from the TRAINING period only, applied to
    both splits — computing them over the full sample would leak.
    """
    df = panel.merge(prices_feat, on=["date", "ticker"], how="inner")
    df["rsi_14"] = df["rsi_14"] / 100.0
    df["fwd_ret"] = df.groupby("ticker")["adj_close"].shift(-horizon) \
        / df["adj_close"] - 1.0
    df = df.dropna(subset=FEATURES + ["fwd_ret"])

    oos = pd.Timestamp(BACKTEST.oos_start)
    train_stats = df[df["date"] < oos][FEATURES].agg(["mean", "std"])
    for f in FEATURES:
        df[f] = (df[f] - train_stats.loc["mean", f]) \
            / (train_stats.loc["std", f] + 1e-9)

    X, y, meta = [], [], []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[FEATURES].to_numpy(dtype=np.float32)
        for i in range(SEQ_LEN, len(g)):
            X.append(vals[i - SEQ_LEN:i])
            y.append(1.0 if g.loc[i, "fwd_ret"] > 0 else 0.0)
            meta.append((g.loc[i, "date"], g.loc[i, "ticker"]))
    X = np.stack(X); y = np.array(y, dtype=np.float32)
    meta = pd.DataFrame(meta, columns=["date", "ticker"])

    train_mask = (meta["date"] < oos).to_numpy()
    return (X[train_mask], y[train_mask], meta[train_mask],
            X[~train_mask], y[~train_mask], meta[~train_mask])


def build_model(n_features: int = len(FEATURES)):
    """Two-layer LSTM classifier (Keras)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), "accuracy"])
    return model


def train(panel: pd.DataFrame, prices_feat: pd.DataFrame,
          horizon: int = 5, epochs: int = 30, batch_size: int = 256):
    """Train with chronological split + early stopping; return model, history,
    and out-of-sample predictions merged with meta for signal generation."""
    from tensorflow import keras

    Xtr, ytr, mtr, Xte, yte, mte = build_dataset(panel, prices_feat, horizon)
    model = build_model()
    hist = model.fit(
        Xtr, ytr, validation_split=0.15, epochs=epochs,
        batch_size=batch_size, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5,
            restore_best_weights=True)],
    )
    preds = mte.assign(p_up=model.predict(Xte, verbose=0).ravel(), y=yte)
    return model, hist, preds
