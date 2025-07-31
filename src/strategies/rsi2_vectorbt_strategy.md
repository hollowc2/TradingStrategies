# 📈 RSI(2) Mean-Reversion Strategy with Vectorbt and MongoDB

## Strategy Overview

- **Universe**: S&P 500 stocks
- **Trend Filter**: Price > 200-day moving average (MA)
- **Entry Signal**: RSI(2) < 10
- **Exit Signal**: Price > 10-day moving average
- **Position Sizing**:
  - Max 10 open positions
  - 10% capital per position
  - If >10 signals, choose the 10 stocks with the lowest RSI(2)

---

## 🧰 Tools and Libraries

- [vectorbt](https://github.com/polakowo/vectorbt) – Backtesting and portfolio analysis
- [pymongo](https://pymongo.readthedocs.io) – MongoDB client for Python
- [pandas](https://pandas.pydata.org/) – Data manipulation
- [NumPy](https://numpy.org/) – Numerical computation

### Install dependencies:

```bash
pip install vectorbt pymongo pandas numpy
```

---

## 💠 Step-by-Step Implementation

### Step 1: Connect to MongoDB and Load Data

Assume you have a MongoDB database named `stock_data` and a collection `daily_stocks`.

```python
from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017")
collection = client["stock_data"]["daily_stocks"]

# Pull all documents and convert to DataFrame
cursor = collection.find({}, {"_id": 0, "symbol": 1, "date": 1, "close": 1})
df = pd.DataFrame(list(cursor))

# Ensure proper types
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])
```

---

### Step 2: Pivot to Wide Format

```python
# Create wide-format close price DataFrame
close_df = df.pivot(index='date', columns='symbol', values='close').sort_index()
```

---

### Step 3: Calculate Indicators

```python
import vectorbt as vbt

rsi = vbt.RSI.run(close_df, window=2)
ma_10 = vbt.MA.run(close_df, window=10)
ma_200 = vbt.MA.run(close_df, window=200)
```

---

### Step 4: Generate Signals

```python
entries = (rsi.rsi < 10) & (close_df > ma_200.ma)
exits = close_df > ma_10.ma
```

---

### Step 5: Apply Universe Filter

You should define your S&P 500 symbols list (from a static list or metadata in your DB).

```python
sp500_symbols = [...]  # List of S&P 500 symbols
close_df = close_df[sp500_symbols]
entries = entries[sp500_symbols]
exits = exits[sp500_symbols]
```

---

### Step 6: Enforce Max 10 Positions Per Day

```python
import numpy as np

# Create an empty mask to hold the final entry signals
final_entries = entries.copy().astype(bool) & False

for date in entries.index:
    daily_signals = entries.loc[date]
    if daily_signals.sum() > 0:
        selected = daily_signals[daily_signals].nsmallest(10, rsi.rsi.loc[date])
        final_entries.loc[date, selected.index] = True
```

---

### Step 7: Create Portfolio

```python
portfolio = vbt.Portfolio.from_signals(
    close=close_df,
    entries=final_entries,
    exits=exits,
    size=0.1,  # 10% of portfolio per position
    init_cash=100_000,
    freq='1D'
)
```

---

### Step 8: Analyze Results

```python
# Performance summary
print(portfolio.stats())

# Equity curve
portfolio.total_return().vbt.plot()
```

---

## 📝 Notes

- Assumes only long positions
- Assumes you don’t re-enter until after exit
- Doesn’t account for transaction costs (can be added via `fees` parameter)
- Ensure your data is cleaned (no duplicate dates, NaNs)

---

## 📌 Optional Enhancements

- Add transaction fees and slippage
- Include volume/liquidity filters
- Store backtest results back to MongoDB
- Evaluate with walk-forward analysis

