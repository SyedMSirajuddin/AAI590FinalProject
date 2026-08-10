# News Sentiment as a Trading Signal: Measuring Predictive Decay Across Holding Horizons

This project is a part of the AAI-590 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status: Completed**

A point-in-time, cost-aware evaluation of FinBERT news sentiment as a multi-horizon equity trading signal. The central question is not merely whether news sentiment predicts returns, but *for how long* any such signal remains exploitable once realistic transaction costs are applied.

---

## Installation

The project is organized as a Python package (`src/`) with a sequence of Jupyter notebooks that drive the analysis end to end.

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/news-sentiment-trading-signal.git
cd news-sentiment-trading-signal
```

**2. Create the environment and install dependencies**

```bash
conda create -n sentiment-capstone python=3.11
conda activate sentiment-capstone
pip install -r requirements.txt
```

Core dependencies include `pandas`, `numpy`, `scikit-learn`, `transformers`, `torch`, `tensorflow`, `datasets`, `yfinance`, `requests`, `matplotlib`, and `jupyter`.

**3. Provide an Alpha Vantage API key**

News and sentiment data are pulled from Alpha Vantage. Create a `.env` file in the project root:

```
ALPHAVANTAGE_API_KEY=your_key_here
```

A free key works but is rate-limited (25 requests/day); a paid tier (75 requests/minute) is recommended for the full multi-year backfill. Prices are pulled from Yahoo Finance via `yfinance` and require no key.

**4. Run the notebooks in order**

Use **Kernel → Restart & Run All** for each, in this sequence:

```
01_data_acquisition_and_cleaning   →  prices_clean.parquet, news_clean.parquet
03_sentiment_model_finetuning      →  news_scored.parquet   (FinBERT scores)
02_exploratory_data_analysis       →  EDA figures
04_signals_and_backtesting         →  baseline multi-horizon backtest
05_optimization_and_analysis       →  tuned backtest + LSTM
06_alphavantage_benchmark          →  FinBERT vs. Alpha Vantage benchmark
07_source_robustness               →  robustness: signal driven by each source
```

Notebook 03 runs before 02 and 04 because it produces the scored news file that the exploratory and backtesting notebooks consume. All downstream stages read `data/processed/news_scored.parquet`, so the news source can be changed without touching the analysis code.

---

## Project Intro / Objective

The main purpose of this project is to determine whether the tone of daily financial news can be converted into a tradeable equity signal, and to characterize how that signal's value changes across holding horizons ranging from one day to roughly six months. Markets generate far more news than any individual investor can read and act on, so a reliable news-sentiment signal would be a valuable decision-support screener. The project treats the holding horizon as the primary experimental variable and evaluates an identical stream of sentiment-derived trades at eight horizons against passive and random-signal baselines, with realistic transaction costs and strict point-in-time discipline. Consistent with sound scientific practice, a rigorously supported negative result is treated as being as valuable as a positive one.

---

## Contributor

- **Syed Sirajuddin** - MS Applied Artificial Intelligence, Shiley-Marcos School of Engineering, University of San Diego. (Individual project.)

---

## Methods Used

- Natural Language Processing (transformer-based financial sentiment classification)
- Machine Learning
- Deep Learning (fine-tuned FinBERT; LSTM sequence model)
- Inferential Statistics (random-signal null hypothesis testing, rank correlation)
- Financial Backtesting and Risk-Adjusted Evaluation
- Data Visualization
- Data Manipulation

---

## Technologies

- Python
- PyTorch and Hugging Face Transformers (FinBERT fine-tuning and scoring)
- TensorFlow / Keras (LSTM sequence model)
- pandas, NumPy, scikit-learn (data manipulation, splits, metrics)
- yfinance (Yahoo Finance price data)
- Alpha Vantage API (news and commercial sentiment)
- Matplotlib (visualization)
- Jupyter Notebook

---

## Project Description

**Overview.** The project builds a complete, sequential pipeline: multi-year news is ingested from Alpha Vantage, each headline is scored by a fine-tuned FinBERT transformer model, scores are aggregated into point-in-time daily trading signals, and a cost-aware backtesting engine evaluates those signals across eight holding horizons against a buy-and-hold baseline and a random-signal null distribution. An independent commercial sentiment score (also from Alpha Vantage) is carried alongside FinBERT's score to benchmark the sentiment measurement and to test the robustness of the final result.

**Datasets.**
- **Prices:** daily open-high-low-close-volume bars for 20 large-capitalization U.S. equities (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, JNJ, XOM, WMT, PG, V, UNH, HD, DIS, NFLX, AMD, BA, PFE), from Yahoo Finance via `yfinance`. 30,140 rows across 1,507 trading sessions (January 2020 – December 2025).
- **News + sentiment:** 100,678 raw articles from the Alpha Vantage market-news-and-sentiment API, reduced to 95,758 after de-duplication and length filtering, spanning March 2022 – December 2025. Each article carries Alpha Vantage's own per-ticker relevance and sentiment scores.
- **Labeled corpus (for the model):** the Financial PhraseBank (Malo et al., 2014), 3,453 expert-annotated financial sentences at the 75% agreement level, split 2,762 / 345 / 346 (train / val / test), used to fine-tune and validate FinBERT.

**Hypothesis.** Daily news sentiment yields a tradeable signal that is strongest at short (swing) horizons and decays as the holding period lengthens.

**Key analyses.** Exploratory analysis of return distributions, cross-ticker correlation (mean 0.361), and the sentiment–return relationship (Spearman ≈ 0.04); FinBERT fine-tuning (test accuracy 0.928, macro-F1 0.901); a FinBERT-vs-Alpha-Vantage sentiment benchmark (75.0% directional agreement); an eight-horizon backtest against a random-signal null; hyperparameter optimization; an LSTM comparison model; and a robustness check that re-runs the strategy on each sentiment source independently.

**Result.** Out-of-sample (all of 2025, tuned on 2022–2024), the strategy beats the random-signal null at zero of eight horizons. Risk-adjusted performance is worst at the short swing horizons (Sharpe −1.30 at one day) and rises toward zero as the horizon lengthens, but never clears the null band. Hyperparameter tuning yields only statistically indistinguishable, near-zero configurations, and the LSTM performs at chance. The signal is "squeezed from both ends": too cost-heavy to exploit at short horizons, too data-sparse to confirm at long ones. The robustness check confirms the same failure using Alpha Vantage's independent sentiment score, showing the negative result does not depend on the particular sentiment model. This is a rigorously supported negative result, consistent with an efficient-market interpretation.

**Roadblocks and challenges.** The primary constraint was news-history depth: an initial free-tier source provided only about one year of news, which was resolved by moving to the Alpha Vantage archive (back to March 2022). Data-quality issues included a ticker-alias mismatch (Alphabet tagged under GOOG, not GOOGL) and a pagination bug that silently truncated heavily-covered tickers, both caught by an added completeness check. A configuration-selection bug and an overly short out-of-sample window in an early evaluation had produced spurious apparent "wins" at two horizons; fixing the split and freezing the selected configuration removed these, underscoring the importance of evaluation discipline.

**Presentation** A video presentation for this project can be found on the following URL: https://youtu.be/cpmL_qIbWHk

**Repository structure.** The repository can be found using the following URL: https://github.com/SyedMSirajuddin/AAI590FinalProject

```
src/
  config.py                     # universe, horizons, backtest settings
  data/                         # prices.py, news.py, news_alphavantage.py, phrasebank.py
  sentiment/                    # finbert.py, finetune.py
  features/                     # aggregate.py, technicals.py
  signals/generate.py           # rule-based signal generation
  backtest/                     # engine.py, baselines.py, metrics.py
  analysis/horizon_decay.py     # horizon table, decay curves, event study
  models/sequence.py            # LSTM comparison model
notebooks/                      # 01 – 07 (run in the order given above)
reports/figures/                # generated figures
data/                           # raw/ and processed/ parquet artifacts (git-ignored)
```

---

## License

This project is released under the MIT License. See the `LICENSE` file in the project root.

---

## Acknowledgments

The author thanks the AAI-590 teaching team at the University of San Diego for their guidance throughout this capstone. Anthropic's Claude was used to assist with code scaffolding and with drafting and revising written sections from the author's own executed pipeline outputs; the author designed the methodology, ran all analyses, and verified all reported results.
