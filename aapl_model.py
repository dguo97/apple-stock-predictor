import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Settings for the risk-advisory module
# ---------------------------------------------------------------------------
# target_vol  : annualised volatility the AAPL position is sized to aim for
# max_weight  : hard cap on the share of the money put into AAPL
# signal_tilt : how far the model's up/down probability may nudge that share
RISK_PROFILES = {
    "Low":    {"target_vol": 0.10, "max_weight": 0.30, "signal_tilt": 0.00},
    "Medium": {"target_vol": 0.18, "max_weight": 0.60, "signal_tilt": 0.05},
    "High":   {"target_vol": 0.30, "max_weight": 1.00, "signal_tilt": 0.10},
}
HORIZON_DAYS = 252          # Monte Carlo projection length (about 1 year)
N_SIMS = 5000               # number of simulated paths
BOOTSTRAP_YEARS = 5         # years of history the simulation resamples from
TRANSACTION_COST = 0.0005   # 0.05% charged on every buy/sell in the backtest

parser = argparse.ArgumentParser(description="AAPL direction model with a risk-aware advisory")
parser.add_argument("--amount", type=float, help="amount to invest in USD (skips the prompt)")
parser.add_argument("--risk", type=str.capitalize, choices=list(RISK_PROFILES),
                    help="risk tolerance: Low, Medium or High (skips the prompt)")
args = parser.parse_args()
if args.amount is not None and args.amount <= 0:
    parser.error("--amount must be a positive number")

# Step 1: Load dataset with yfinance (fetches up to today)
data = yf.download("AAPL", start="1980-12-12", auto_adjust=False, progress=False)
if data.empty:
    sys.exit("No data returned from Yahoo Finance. Check your connection and try again.")

# Newer yfinance versions return MultiIndex columns even for one ticker
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna()
data.index.name = "Date"

# Show first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Step 2: Create target column (1 = adjusted price goes up tomorrow, 0 = goes down/flat)
adj = data['Adj Close']
data['Target'] = (adj.shift(-1) > adj).astype(int)

# Step 3: Feature engineering (stationary features instead of raw prices)
ret_1 = adj.pct_change()
features_df = pd.DataFrame(index=data.index)
features_df['ret_1'] = ret_1
features_df['ret_2'] = ret_1.shift(1)
features_df['ret_3'] = ret_1.shift(2)
features_df['ret_5d'] = adj.pct_change(5)
features_df['ret_10d'] = adj.pct_change(10)
features_df['ma5_ratio'] = adj / adj.rolling(5).mean() - 1
features_df['ma20_ratio'] = adj / adj.rolling(20).mean() - 1
features_df['ma50_ratio'] = adj / adj.rolling(50).mean() - 1
features_df['volatility_10'] = ret_1.rolling(10).std()
features_df['hl_range'] = (data['High'] - data['Low']) / data['Close']
features_df['oc_change'] = (data['Close'] - data['Open']) / data['Open']
features_df['volume_change'] = data['Volume'].pct_change().replace([np.inf, -np.inf], np.nan)

# RSI (14-day)
delta = adj.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
features_df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

features = list(features_df.columns)
model_data = features_df.join(data['Target']).iloc[:-1]  # last row has no "tomorrow"
model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()
X = model_data[features]
y = model_data['Target']

# Step 4: Train-test split (chronological, no shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=20, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test)

# Step 7: Evaluation, with a majority-class baseline and time-series cross-validation
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)

print("\nBaseline (always predict majority class) accuracy:", round(baseline_acc, 4))
print("Model accuracy on hold-out set:", round(accuracy_score(y_test, predictions), 4))
print(f"TimeSeriesSplit CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))

# Step 8: Risk & Investment Advisory (the amount and risk tolerance drive everything below)
print("\n--- Apple (AAPL) Stock Investment Risk Advisory ---")
latest_price = data['Close'].iloc[-1]
annualized_volatility = ret_1.std() * np.sqrt(252) * 100
recent_vol = ret_1.tail(252).std() * np.sqrt(252)  # last ~12 months, used for sizing

print(f"Latest Available Closing Price: ${latest_price:.2f}")
print(f"Historical Annualized Volatility: {annualized_volatility:.2f}% "
      f"(last 12 months: {recent_vol * 100:.2f}%)")


def ask_amount():
    if args.amount is not None:
        return args.amount
    if not sys.stdin.isatty():
        print("\nNon-interactive session detected; using $1,000 as the amount.")
        return 1000.0
    while True:
        try:
            value = float(input("\nEnter amount you plan to invest ($): ").replace(",", ""))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive number.")


def ask_risk():
    if args.risk is not None:
        return args.risk
    if not sys.stdin.isatty():
        print("Non-interactive session detected; using Medium risk tolerance.")
        return "Medium"
    while True:
        value = input("Enter your risk tolerance (Low / Medium / High): ").strip().capitalize()
        if value in RISK_PROFILES:
            return value
        print("Please type Low, Medium or High.")


investment_amount = ask_amount()
risk_tolerance = ask_risk()
profile = RISK_PROFILES[risk_tolerance]

# 8a. Model signal for the next trading day
latest_features = features_df.iloc[[-1]]
p_up = 0.5
if not latest_features.isna().any().any() and 1 in list(model.classes_):
    p_up = float(model.predict_proba(latest_features)[0, list(model.classes_).index(1)])

# 8b. Position sizing: target a volatility level, cap the weight, nudge by the signal
base_weight = min(profile["max_weight"], profile["target_vol"] / recent_vol)
weight = float(np.clip(base_weight + profile["signal_tilt"] * (2 * p_up - 1),
                       0.0, profile["max_weight"]))
stock_dollars = investment_amount * weight
cash_dollars = investment_amount - stock_dollars
shares = stock_dollars / latest_price

# 8c. Monte Carlo: resample recent daily returns to project the value of the split portfolio
history = ret_1.dropna().tail(252 * BOOTSTRAP_YEARS).to_numpy()
rng = np.random.default_rng(42)
sampled = rng.choice(history, size=(N_SIMS, HORIZON_DAYS), replace=True)
total_paths = cash_dollars + stock_dollars * np.cumprod(1 + sampled, axis=1)
total_paths = np.hstack([np.full((N_SIMS, 1), investment_amount), total_paths])
final_values = total_paths[:, -1]

p5, p50, p95 = np.percentile(final_values, [5, 50, 95])
prob_loss = float((final_values < investment_amount).mean())
var_95 = max(0.0, investment_amount - p5)
cvar_95 = max(0.0, investment_amount - final_values[final_values <= p5].mean())
drawdowns = (total_paths / np.maximum.accumulate(total_paths, axis=1) - 1).min(axis=1)
bad_drawdown = float(np.percentile(drawdowns, 5))


# 8d. Backtest on the hold-out period: what would this amount have done?
def summarise(daily_returns):
    equity = investment_amount * (1 + daily_returns).cumprod()
    years = len(daily_returns) / 252
    std = daily_returns.std()
    return {
        "final": equity.iloc[-1],
        "total_return": equity.iloc[-1] / investment_amount - 1,
        "ann_vol": std * np.sqrt(252),
        "sharpe": (daily_returns.mean() / std * np.sqrt(252)) if std > 0 else float("nan"),
        "max_dd": (equity / equity.cummax() - 1).min(),
        "years": years,
    }


next_ret = ret_1.shift(-1).reindex(X_test.index)
signal = pd.Series(predictions, index=X_test.index)
switches = signal.diff().abs()
switches.iloc[0] = signal.iloc[0]  # cost of the initial purchase
strategies = pd.DataFrame({
    "Buy & hold (100% AAPL)": next_ret,
    f"{risk_tolerance} allocation ({weight:.0%} AAPL)": weight * next_ret,
    "Model signal (AAPL or cash)": signal * next_ret - switches * TRANSACTION_COST,
}).dropna()
equity_curves = investment_amount * (1 + strategies).cumprod()
backtest = {name: summarise(strategies[name]) for name in strategies}

print("\n" + "=" * 62)
print("INVESTMENT RISK ASSESSMENT")
print("=" * 62)
print(f"Proposed investment : ${investment_amount:,.2f}")
print(f"Risk profile        : {risk_tolerance} "
      f"(target volatility {profile['target_vol']:.0%}, max {profile['max_weight']:.0%} in AAPL)")
if profile["signal_tilt"] > 0:
    tilt_note = f"the model is weak, so it only nudges the AAPL weight by up to {profile['signal_tilt']:.0%}"
else:
    tilt_note = "the model is weak, so this profile ignores it when sizing"
print(f"Model P(up) tomorrow: {p_up:.1%}  ({tilt_note})")

print("\nSuggested split")
print(f"  AAPL : ${stock_dollars:,.2f}  ({weight:.1%})  ~ {shares:,.3f} shares at ${latest_price:.2f}")
print(f"  Cash / diversified low-risk assets : ${cash_dollars:,.2f}  ({1 - weight:.1%})")

print(f"\nSimulated outcomes over {HORIZON_DAYS} trading days ({N_SIMS:,} paths)")
print(f"  Median value          : ${p50:,.2f}")
print(f"  Likely range (5%-95%) : ${p5:,.2f} to ${p95:,.2f}")
print(f"  Chance of ending below ${investment_amount:,.0f}: {prob_loss:.1%}")
print(f"  Value at Risk (95%)   : ${var_95:,.2f}  (loss exceeded in 1 of 20 outcomes)")
print(f"  Expected shortfall    : ${cvar_95:,.2f}  (average loss in the worst 5%)")
print(f"  Bad-case drawdown     : {bad_drawdown:.1%}  (peak-to-trough, 1 in 20 paths is worse)")

first_day, last_day = strategies.index[0].date(), strategies.index[-1].date()
print(f"\nBacktest of ${investment_amount:,.0f} on the hold-out period ({first_day} to {last_day})")
print(f"  {'Strategy':<34}{'Final value':>14}{'Return':>10}{'Volatility':>12}{'Sharpe':>8}{'Max DD':>9}")
for name, m in backtest.items():
    print(f"  {name:<34}${m['final']:>13,.2f}{m['total_return']:>10.1%}"
          f"{m['ann_vol']:>12.1%}{m['sharpe']:>8.2f}{m['max_dd']:>9.1%}")
print(f"  (model signal includes a {TRANSACTION_COST:.2%} cost per trade; no taxes, no slippage)")

if risk_tolerance == "Low":
    print("\nWARNING: Equities like Apple carry significant market and volatility risk. A low risk tolerance suggests exploring diversified index funds or fixed-income assets instead of individual stock picking.")
elif risk_tolerance == "Medium":
    print("\nCAUTION: Individual stocks can experience sharp drawdowns. Ensure this investment represents a well-diversified portion of your overall portfolio.")
else:
    print("\nNOTICE: High risk tolerance accepted, but remember that machine learning models and historical trends do not protect against sudden market downturns.")
print("Simulation and backtest use past returns, which may not repeat. This is not financial advice.")
print("=" * 62 + "\n")

# Step 9: Charts for the advisory (projection fan chart and backtest equity curves)
days = np.arange(HORIZON_DAYS + 1)
bands = np.percentile(total_paths, [5, 25, 50, 75, 95], axis=0)
plt.figure(figsize=(12, 4))
plt.fill_between(days, bands[0], bands[4], alpha=0.15, color='tab:blue', label='5%-95%')
plt.fill_between(days, bands[1], bands[3], alpha=0.30, color='tab:blue', label='25%-75%')
plt.plot(days, bands[2], color='tab:blue', label='Median')
plt.axhline(investment_amount, color='gray', linestyle='--', label='Amount invested')
plt.title(f"Simulated value of ${investment_amount:,.0f} ({weight:.0%} AAPL, {risk_tolerance} risk)")
plt.xlabel("Trading days from today")
plt.ylabel("Portfolio value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
for name in equity_curves:
    plt.plot(equity_curves.index, equity_curves[name], label=name)
plt.axhline(investment_amount, color='gray', linestyle='--')
plt.title(f"Hold-out backtest of ${investment_amount:,.0f}")
plt.xlabel("Date")
plt.ylabel("Portfolio value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 10: Plot adjusted closing prices
plt.figure(figsize=(12, 4))
plt.plot(adj, label='AAPL Adj Close Price')
plt.title('Apple Stock Adjusted Closing Price Over Time')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

# Step 12: Feature Importance
importances = model.feature_importances_
order = np.argsort(importances)
plt.figure(figsize=(8, 5))
plt.barh(np.array(features)[order], importances[order], color='green')
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Step 13: Interactive Candlestick Chart using Plotly
print("\nOpening interactive candlestick chart in browser window...")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

fig.update_layout(
    title="Apple Stock Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

fig.show()
