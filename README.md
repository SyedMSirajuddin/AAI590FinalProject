# News Sentiment as a Trading Signal: Measuring Predictive Decay Across Holding Horizons

This project is a part of the AAI-590 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

-- Project Status: Active

## Installation

To use this project, first clone the repo on your device:

```bash
git clone https://github.com/<your-username>/sentiment-horizon-capstone.git
cd sentiment-horizon-capstone
pip install -r requirements.txt
```

To run the full pipeline against live data you need a free Finnhub API key:

```bash
export FINNHUB_API_KEY=your_key_here
python scripts/run_pipeline.py
```

To verify the machinery without any API keys or GPU (synthetic-data mode with a planted, decaying sentiment effect):

```bash
python scripts/demo_synthetic.py
python scripts/run_eda.py --synthetic
streamlit run scripts/dashboard.py
```

Fine-tuning the sentiment model on Financial PhraseBank (GPU recommended):

```bash
python -m src.sentiment.finetune --base bert-base-uncased
```

## Project Intro/Objective

The main purpose of this project is to determine whether daily financial-news sentiment, combined with historical price behavior, can be converted into trading signals that reliably identify profitable opportunities — and, critically, to measure **how the predictive value of those signals decays as the holding horizon lengthens** from days to weeks to months. Rather than committing to a single trading style, the holding horizon is treated as the primary experimental variable: the same signal stream is backtested at horizons from 1 to 126 trading days against buy-and-hold and random-signal (no-skill) baselines. The end user is a retail investor seeking a research and decision-support screener, not a black-box autopilot. A finding that the signal disappears beyond a certain horizon is as valuable as a finding that it persists.

## Partner(s)/Contributor(s)

- Syed Sirajuddin

## Methods Used

- NLP (transformer-based financial sentiment analysis)
- Deep Learning (FinBERT fine-tuning; optional LSTM signal combiner)
- Machine Learning
- Time-Series Backtesting & Event Studies
- Inferential Statistics (Monte-Carlo null distributions)
- Data Visualization
- Data Manipulation

## Technologies

- Python
- PyTorch / Hugging Face Transformers & Datasets
- TensorFlow/Keras (optional LSTM model)
- pandas, NumPy, scikit-learn, Matplotlib, PyArrow
- yfinance, Finnhub API, GDELT DOC API
- Streamlit

## Project Description

Two point-in-time-aligned data streams are used: (1) daily OHLCV bars for a 20-name universe of liquid U.S. large-caps (2020–2025, via yfinance), and (2) historical financial news headlines per ticker (Finnhub `/company-news`, with GDELT as a long-history fallback), plus the Financial PhraseBank corpus (~4,800 labeled sentences) for fine-tuning and validating the sentiment model.

The pipeline: news ingestion and cleaning → FinBERT sentiment scoring (scalar score = P(pos) − P(neg)) → daily per-ticker aggregation with confidence weighting and rolling windows → rule-based signal generation with a trend filter → a fixed-horizon backtest engine that evaluates the *same* signals at 1, 3, 5, 10, 21, 42, 63, and 126 trading-day holds with a one-bar execution delay and transaction costs → comparison against buy-and-hold and a Monte-Carlo random-signal null → decay-curve, event-study, and equity-curve visualizations. Look-ahead bias is controlled by mapping every article to the first trading session on which it could have been acted upon, applying a strictly chronological out-of-sample split, and normalizing model features with training-period statistics only.

Central hypothesis: aggregated daily news sentiment carries predictive value that is strongest over short (swing) horizons and decays as the holding period lengthens. Key challenges include news-to-ticker linkage noise, survivorship/look-ahead bias, transaction-cost drag at short horizons, and limited free-tier news history for long-horizon tests.

### Repository structure

| Path | Capstone element | Contents |
| --- | --- | --- |
| `src/data/prices.py` | Data Cleaning | OHLCV download + documented cleaning steps |
| `src/data/news.py` | Data Cleaning | Finnhub/GDELT fetchers, dedup, point-in-time alignment |
| `src/data/phrasebank.py` | Data Cleaning | Financial PhraseBank loader + stratified splits |
| `scripts/run_eda.py` | Exploratory Data Analysis | Coverage, distributions, correlations, sentiment-vs-return |
| `src/sentiment/`, `src/features/`, `src/signals/` | Model/Pipeline Design & Building | FinBERT scorer, aggregation, signal rules |
| `src/sentiment/finetune.py`, `src/models/sequence.py` | Model Training | PhraseBank fine-tuning; optional LSTM combiner |
| `src/config.py`, threshold/window params | Model Optimization | Centralized hyperparameters; tune in-sample, report OOS |
| `src/backtest/`, `src/analysis/` | Model/Pipeline Analysis & Discussion | Horizon engine, baselines, metrics, decay analysis |
| `scripts/run_pipeline.py` | — | End-to-end orchestration |
| `scripts/demo_synthetic.py` | — | Pipeline validation on a planted, decaying effect |
| `scripts/dashboard.py` | — | Streamlit screener (end-user deliverable) |

### Pipeline validation

`demo_synthetic.py` constructs a synthetic market in which news sentiment has genuine predictive power that decays geometrically over ~5 trading days. The full pipeline recovers this planted effect — out-of-sample Sharpe peaks at the 3–5 day horizons and falls inside the random-signal null band beyond ~10 days — which validates the backtest engine, cost model, and decay analysis before any live data is used.

### Notebooks

The five notebooks under `notebooks/` narrate the full project for a semi-technical reader and map one-to-one onto the required code-base elements and report sections (the mapping table is in Notebook 01). Run order: 01 2192 03 2192 02 2192 04 2192 05.

## License

MIT — see `LICENSE`.

## Acknowledgments

Thanks to the AAI-590 instructors at the University of San Diego for guidance, and to the authors of FinBERT (Araci, 2019; ProsusAI) and the Financial PhraseBank (Malo et al., 2014).
