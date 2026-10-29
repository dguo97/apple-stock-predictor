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
- Risk advisory driven by your **investment amount** and **risk tolerance** (Low / Medium / High):
  - volatility-targeted position sizing: how much goes into AAPL and how much stays in cash, in dollars and shares
  - Monte Carlo projection over one year: median and 5%-95% range, chance of loss, Value at Risk, expected shortfall and bad-case drawdown
  - hold-out backtest of your amount: buy and hold vs. your risk-profile allocation vs. the model's up/down signal (with trading costs)
  - fan chart and equity-curve chart

## Getting started

```bash
git clone https://github.com/dguo97/aapl-stock-predictor.git
cd aapl-stock-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python aapl_model.py
```

The script asks for the amount and risk tolerance. To skip the prompts (or run it non-interactively), pass them as options:

```bash
python aapl_model.py --amount 10000 --risk Medium
```

With no terminal and no options it defaults to $1,000 and Medium risk.

## How it works

1. **Load** daily AAPL data (Open, High, Low, Close, Adj Close, Volume).
2. **Target:** `1` if the next day's adjusted close is higher than today's, otherwise `0`.
3. **Features:** built only from information available on the day, so nothing from the future leaks in. The final row is dropped because it has no "tomorrow".
4. **Train/test:** the first 80% of dates train the model and the final 20% test it.
5. **Evaluate:** hold-out accuracy, a majority-class baseline, and 5-fold `TimeSeriesSplit` cross-validation.
6. **Advise:** your amount and risk tolerance are used as follows.
   - *Sizing:* the AAPL weight is the profile's target volatility divided by AAPL's last-12-month volatility, capped per profile (Low 30%, Medium 60%, High 100%). The model's probability of an up day nudges it by at most 0% / 5% / 10%, because the model is weak.
   - *Projection:* 5,000 one-year paths are built by resampling the last 5 years of daily returns, applied to the AAPL part of your money (the rest stays flat).
   - *Backtest:* the same amount is run through the hold-out period three ways, with a 0.05% cost per trade for the model-signal strategy.
7. **Visualise:** projection fan chart, backtest curves, price chart, confusion matrix, feature importances, candlestick chart.

The risk profiles, horizon, number of simulations and trading cost are constants at the top of `aapl_model.py`. The simulation resamples past returns, so it assumes the future resembles recent history; real losses can be larger.

## Project structure

```
aapl-stock-predictor/
├── aapl_model.py      # data loading, features, model, evaluation, charts
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Ideas for improvement

- Add more features (MACD, Bollinger bands, market index or sector returns)
- Try other models (gradient boosting, logistic regression) and compare against the baseline
- Split the script into modules (`data.py`, `features.py`, `model.py`) and add tests

## License

MIT. See `LICENSE`.
