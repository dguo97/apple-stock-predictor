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

# Step 8: Interactive Risk & Investment Advisory Module
print("\n--- Apple (AAPL) Stock Investment Risk Advisory ---")
latest_price = data['Close'].iloc[-1]
annualized_volatility = adj.pct_change().std() * np.sqrt(252) * 100

print(f"Latest Available Closing Price: ${latest_price:.2f}")
print(f"Historical Annualized Volatility: {annualized_volatility:.2f}%")

if sys.stdin.isatty():
    try:
        investment_amount = float(input("\nEnter amount you plan to invest ($): "))
    except ValueError:
        print("Invalid amount entered; using $1,000.00 as a placeholder.")
        investment_amount = 1000.0
    risk_tolerance = input("Enter your risk tolerance (Low / Medium / High): ").strip().capitalize()
    if risk_tolerance not in ("Low", "Medium", "High"):
        print("Unrecognised risk tolerance; defaulting to Medium.")
        risk_tolerance = "Medium"
else:
    print("\nNon-interactive session detected; using defaults ($1,000, Medium).")
    investment_amount, risk_tolerance = 1000.0, "Medium"

print("\n" + "=" * 50)
print("INVESTMENT RISK ASSESSMENT:")
print(f"- Proposed Investment: ${investment_amount:,.2f}")
print(f"- Risk Profile: {risk_tolerance}")

if risk_tolerance == "Low":
    print("WARNING: Equities like Apple carry significant market and volatility risk. A low risk tolerance suggests exploring diversified index funds or fixed-income assets instead of individual stock picking.")
elif risk_tolerance == "Medium":
    print("CAUTION: Individual stocks can experience sharp drawdowns. Ensure this investment represents a well-diversified portion of your overall portfolio.")
else:
    print("NOTICE: High risk tolerance accepted, but remember that machine learning models and historical trends do not protect against sudden market downturns.")
print("=" * 50 + "\n")

# Step 9: Plot adjusted closing prices
plt.figure(figsize=(12, 4))
plt.plot(adj, label='AAPL Adj Close Price')
plt.title('Apple Stock Adjusted Closing Price Over Time')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

# Step 11: Feature Importance
importances = model.feature_importances_
order = np.argsort(importances)
plt.figure(figsize=(8, 5))
plt.barh(np.array(features)[order], importances[order], color='green')
plt.title("Feature Importance in Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Step 12: Interactive Candlestick Chart using Plotly
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
