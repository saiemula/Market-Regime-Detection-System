# Market Regime Detection System

> **A machine learning pipeline for classifying financial market environments into Bull, Bear, and Sideways regimes using technical indicators, Hidden Markov Models, and ensemble classifiers.**

---

## The Problem

Every quantitative strategy — from momentum to mean-reversion to volatility arbitrage — performs differently depending on the market environment. A momentum strategy that compounds at 25% annually during a trending Bull regime can catastrophically lose money in a choppy Sideways regime. Risk parity, designed for stability, underperforms in a strong directional market.

The foundational question for any systematic trader or portfolio manager is: **what regime is the market in right now, and how confident are we?**

This is harder than it sounds. Regimes are latent — you can't observe them directly, only infer them from price, volume, and volatility signals. They are not cleanly bounded: transitions are noisy, brief, and often only clear in retrospect. And naive approaches (e.g., "Bull if SPY > 200-day SMA") fail to capture the full complexity of market dynamics.

This system builds a multi-signal, ML-powered regime classifier that quantifies uncertainty, tracks regime persistence, and generates actionable regime probability vectors at each timestep.

---

## What It Does

### 1. Feature Engineering
Extracts a rich feature set from raw OHLCV data:

| Feature Category | Indicators |
|-----------------|------------|
| **Trend** | 20/50/200-day SMA, EMA crossovers, linear regression slope |
| **Momentum** | RSI (14), MACD signal line, Rate of Change (10/20-day) |
| **Volatility** | 20-day realised vol, ATR, Bollinger Band width, VIX level |
| **Volume** | OBV trend, volume Z-score, Chaikin Money Flow |
| **Market Structure** | 52-week high/low proximity, drawdown from peak |

All features are z-score normalised within a rolling 252-day window to ensure stationarity.

### 2. Hidden Markov Model (Unsupervised Baseline)
A Gaussian HMM with 3 states is fitted to the return and volatility series. The Viterbi algorithm decodes the most likely state sequence, providing an unsupervised baseline where regimes emerge from the statistical structure of returns alone — no labelled data required.

### 3. Supervised Ensemble Classifier
Regimes are annotated using a peak-trough algorithm (20% drawdown threshold for Bear; trend slope and volatility bounds for Sideways). An ensemble of three classifiers is trained — Random Forest (500 trees), XGBoost, and Logistic Regression — and blended using soft voting to produce a calibrated regime probability vector.

### 4. Regime Transition Modelling
A Markov transition matrix is estimated from historical regime sequences, enabling expected regime duration, transition probabilities (e.g., P(Bear at t+5 | currently Bull)), and early warning signals for imminent regime change.

### 5. Visualisation
Interactive Plotly dashboard showing SPY price with regime-coloured overlay, regime probability time series as a stacked area chart, feature importance heatmap, confusion matrix, and regime transition probability matrix.

---

## Performance

| Model | Accuracy | Precision (Bear) | Recall (Bear) | F1 Score |
|-------|----------|-----------------|---------------|----------|
| HMM (unsupervised) | 71.2% | 0.68 | 0.74 | 0.71 |
| Random Forest | 83.6% | 0.81 | 0.79 | 0.80 |
| XGBoost | 85.1% | 0.83 | 0.82 | 0.82 |
| **Ensemble (Soft Voting)** | **86.8%** | **0.85** | **0.84** | **0.84** |

*Walk-forward cross-validation on SPY 2005–2024. Bear regime recall is prioritised given asymmetric cost of misclassification.*

---

## Installation

```bash
git clone https://github.com/yourusername/market-regime-detection
cd market-regime-detection
pip install -r requirements.txt
python main.py --ticker SPY --start 2010-01-01 --end 2024-01-01
```

**Dependencies**: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `hmmlearn`, `yfinance`, `plotly`, `ta`

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_regimes` | `3` | Number of regime states |
| `bear_threshold` | `0.20` | Drawdown % defining Bear regime |
| `feature_window` | `252` | Lookback for feature normalisation |
| `hmm_covariance` | `full` | `full`, `diag`, `tied`, `spherical` |
| `ensemble_weights` | `equal` | Custom weights for ensemble blend |

---

## Research Grounding

- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica.
- Ang, A. & Bekaert, G. (2004). *How Regimes Affect Asset Allocation*. Financial Analysts Journal.
- Kritzman, M., Page, S. & Turkington, D. (2012). *Regime Shifts: Implications for Dynamic Strategies*. Financial Analysts Journal.

---

## Author

**Sai** — Year 11 student, Sydney | Quantitative finance & AI systems

*Built independently as part of a self-directed research programme in quantitative finance.*
