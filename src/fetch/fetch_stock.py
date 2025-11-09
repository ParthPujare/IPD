"""
Fetch and update historical + latest stock prices for ADANIGREEN.NS using yfinance.
Keeps both 'stock_data.csv' and 'features_enhanced.csv' in sync for model predictions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# --- Path setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir

# --- Constants ---
TICKER = "ADANIGREEN.NS"
ROOT = get_project_root()
STOCK_DATA_PATH = ROOT / "data" / "stock_data.csv"
FEATURES_DATA_PATH = ROOT / "data" / "features_enhanced.csv"


# ==========================================================
# 1. Fetch recent data
# ==========================================================
def fetch_stock_data(period="6mo"):
    try:
        print(f" Fetching {period} of stock data for {TICKER} from Yahoo Finance...")
        ticker = yf.Ticker(TICKER)
        df = ticker.history(period=period, interval="1d")

        if df.empty:
            raise ValueError(f"No data fetched for {TICKER}")

        df.reset_index(inplace=True)
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = TICKER

        # All lowercase for consistency
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        print(f" Fetched {len(df)} new records for {TICKER}")
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f" Error fetching stock data: {e}")
        raise


# ==========================================================
# 2. Update stock_data.csv
# ==========================================================
def update_stock_data():
    ensure_dir(STOCK_DATA_PATH.parent)
    new_df = fetch_stock_data(period="6mo")

    if STOCK_DATA_PATH.exists():
        try:
            existing_df = pd.read_csv(STOCK_DATA_PATH)
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
            combined_df = combined_df.sort_values("date").reset_index(drop=True)
            print(f" Updated stock data: {len(existing_df)} → {len(combined_df)} records")
        except Exception as e:
            print(f" Error reading existing stock data, using new only: {e}")
            combined_df = new_df
    else:
        combined_df = new_df
        print(f" Created new stock data file with {len(combined_df)} records")

    combined_df.to_csv(STOCK_DATA_PATH, index=False)
    print(f" Stock data saved to {STOCK_DATA_PATH}")
    return combined_df


# ==========================================================
# 3. Feature engineering
# ==========================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Moving averages and exponential MAs ---
    for window in [5, 10, 20]:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()

    # --- RSI ---
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- Returns & volatility ---
    df["returns_1d"] = df["close"].pct_change()
    df["returns_5d"] = df["close"].pct_change(5)
    df["returns_10d"] = df["close"].pct_change(10)
    df["volatility_5d"] = df["returns_1d"].rolling(5).std()
    df["volatility_20d"] = df["returns_1d"].rolling(20).std()

    # --- Price spreads ---
    df["high_low_pct"] = (df["high"] - df["low"]) / df["low"]
    df["volume_sma_5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_5"] + 1e-9)

    # --- Lagged Close prices & returns ---
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"close_ret_{lag}d"] = df["close"].pct_change(lag)

    # --- Other derived features ---
    df["open_ret_1d"] = df["open"].pct_change()
    df["high_ret_1d"] = df["high"].pct_change()
    df["low_ret_1d"] = df["low"].pct_change()

    # --- Log & rolling metrics ---
    df["close_log"] = np.log(df["close"] + 1e-9)
    for w in [3, 5, 10, 20]:
        df[f"close_sma_{w}"] = df["close"].rolling(w).mean()
        df[f"close_std_{w}"] = df["close"].rolling(w).std()
        df[f"close_vol_{w}"] = df[f"close_std_{w}"] / (df[f"close_sma_{w}"] + 1e-9)

    # --- Time-based features ---
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # --- Momentum & returns ---
    df["return_1d"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["momentum_3d"] = df["close"] / df["close"].shift(3) - 1

    # --- Sentiment placeholders ---
    df["sentiment_score_mean"] = 0.0
    df["sentiment_score_std"] = 0.0
    df["sentiment_count"] = 0

    df = df.bfill().ffill().fillna(0)
    return df


# ==========================================================
# 4. Update features_enhanced.csv
# ==========================================================
def update_features_enhanced():
    ensure_dir(FEATURES_DATA_PATH.parent)
    df = update_stock_data()
    if df.empty:
        print(" No data available to update features_enhanced.csv.")
        return None

    print(" Generating enhanced feature dataset (full 56-feature pipeline)...")
    df = compute_features(df)

    df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f" Enhanced features saved to {FEATURES_DATA_PATH}")
    print(f" {len(df)} rows | {len(df.columns)} columns generated")
    return df


# ==========================================================
# 5. Standalone execution
# ==========================================================
if __name__ == "__main__":
    print(" Running stock data updater...")
    df = update_features_enhanced()
    if df is not None:
        print(f"\n Data preview:\n{df.tail(5)}")
        print(f"\n Date range: {df['date'].min()} → {df['date'].max()}")
