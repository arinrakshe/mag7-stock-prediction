# MAG7 Stock Price Prediction

Predicts next-day closing prices for the MAG7 stocks (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA) using XGBoost on engineered technical indicators.

## What it does

1. Downloads 10 years of daily price history (2015–2024) from Yahoo Finance.
2. Builds features per company: daily/log returns, moving averages (7, 30), EMA & MACD, volatility, Bollinger Bands, RSI, momentum, volume ratios, lagged closes.
3. Trains one XGBoost regressor per company on an 80/20 time-ordered split.
4. Reports per-company RMSE and feature importance, and saves four plots.

## Run

```bash
pip install yfinance pandas numpy scikit-learn xgboost matplotlib
python3 finance.py
```

Outputs (gitignored): per-ticker CSVs, `cleaned_stocks.csv`, `engineered_stocks.csv`, `rmse_summary.csv`, `feature_importance.csv`, `predictions.csv`, and PNG plots.
