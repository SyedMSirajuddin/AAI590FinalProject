"""
Central configuration for the sentiment-horizon trading pipeline.

All tunable parameters live here so that experiments are reproducible:
change a value once, and every stage of the pipeline (ingestion, scoring,
signal generation, backtesting) picks it up consistently.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Universe and date range
# ---------------------------------------------------------------------------
@dataclass
class UniverseConfig:
    """Stock universe under study: liquid, large-cap U.S. names with heavy
    news coverage (news-to-ticker linkage is far more reliable for these)."""
    tickers: list = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "JNJ", "XOM",
        "WMT", "PG", "V", "UNH", "HD",
        "DIS", "NFLX", "AMD", "BA", "PFE",
    ])
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    benchmark: str = "SPY"  # market baseline for buy-and-hold comparison


# ---------------------------------------------------------------------------
# Sentiment model
# ---------------------------------------------------------------------------
@dataclass
class SentimentConfig:
    # Pre-trained FinBERT checkpoint (fine-tuned on Financial PhraseBank)
    model_name: str = "ProsusAI/finbert"
    # Local fine-tuning settings (Financial PhraseBank, sentences_75agree)
    finetune_base: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    seed: int = 42


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
@dataclass
class SignalConfig:
    # Rolling window (trading days) over which daily sentiment is aggregated
    sentiment_window: int = 3
    # Minimum number of articles in window for a signal to be considered valid
    min_articles: int = 2
    # Sentiment score threshold (net positive-minus-negative, in [-1, 1])
    long_threshold: float = 0.35
    short_threshold: float = -0.35   # set to None to run long-only
    # Simple price filter: require close above N-day SMA for longs
    trend_filter_sma: int = 20


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    # Holding horizons in trading days — the primary experimental variable.
    # 5d ≈ 1 week (swing), 21d ≈ 1 month, 63d ≈ 1 quarter (position),
    # 126d ≈ 6 months (long horizon).
    horizons: list = field(default_factory=lambda: [1, 3, 5, 10, 21, 42, 63, 126])
    # Round-trip transaction cost, as a fraction (10 bps per side = 0.002 RT)
    cost_per_side: float = 0.001
    # Capital allocation: equal-weight across simultaneous open positions
    max_positions: int = 10
    # Train/test split for out-of-sample evaluation (date boundary)
    oos_start: str = "2024-01-01"
    # Annualization factor for Sharpe (daily returns)
    trading_days: int = 252
    risk_free_rate: float = 0.0


UNIVERSE = UniverseConfig()
SENTIMENT = SentimentConfig()
SIGNALS = SignalConfig()
BACKTEST = BacktestConfig()
