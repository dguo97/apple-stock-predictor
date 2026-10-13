# AAPL Stock Direction Predictor

A small machine-learning project that downloads Apple (AAPL) price history, trains a Random Forest to predict whether the next day's adjusted close will be **up or down**, and shows the results with charts and a simple risk-advisory prompt.

> **Disclaimer:** This is an educational project, not financial advice. Next-day stock direction is very hard to predict, and the model's accuracy is expected to be close to a coin flip. Do not make investment decisions based on it.

## Features

- Downloads full AAPL history up to today with [`yfinance`](https://github.com/ranaroussi/yfinance)
- Engineered, stationary features: lagged returns, moving-average ratios, volatility, high-low range, RSI, volume change
- Chronological train/test split (no shuffling, no look-ahead leakage)
- Evaluation against a majority-class baseline, plus `TimeSeriesSplit` cross-validation
- Plots: adjusted close price, confusion matrix, feature importance
- Interactive Plotly candlestick chart
- Terminal risk-advisory prompt (investment amount and risk tolerance), with safe defaults when run non-interactively

## Getting started

```bash
git clone https://github.com/<your-username>/aapl-stock-predictor.git
cd aapl-stock-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python aapl_model.py
```

## How it works

1. **Load** daily AAPL data (Open, High, Low, Close, Adj Close, Volume).
2. **Target:** `1` if the next day's adjusted close is higher than today's, otherwise `0`.
3. **Features:** built only from information available on the day, so nothing from the future leaks in. The final row is dropped because it has no "tomorrow".
4. **Train/test:** the first 80% of dates train the model and the final 20% test it.
5. **Evaluate:** hold-out accuracy, a majority-class baseline, and 5-fold `TimeSeriesSplit` cross-validation.
6. **Visualise:** price chart, confusion matrix, feature importances, candlestick chart.

## Project structure

```
aapl-stock-predictor/
├── aapl_model.py      # data loading, features, model, evaluation, charts
├── requirements.txt
├── .gitignore
└── README.md
```

## Ideas for improvement

- Add more features (MACD, Bollinger bands, market index or sector returns)
- Try other models (gradient boosting, logistic regression) and compare against the baseline
- Add transaction costs and a simple backtest of a strategy based on the predictions
- Split the script into modules (`data.py`, `features.py`, `model.py`) and add tests
