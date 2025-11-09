"""
Fetch and refresh full historical stock data (2020→today) for ADANIGREEN.NS using yfinance.
Generates aligned enhanced feature dataset compatible with trained models.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

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
# 1. Fetch full data (2020 → today)
# ==========================================================
def fetch_full_stock_data():
    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    print(f"Fetching full stock data for {TICKER} from {start_date} to {end_date} ...")
    ticker = yf.Ticker(TICKER)
    df = ticker.history(start=start_date, end=end_date, interval="1d")

    if df.empty:
        raise ValueError(f"No data fetched for {TICKER}")

    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = TICKER

    print(f"Fetched {len(df)} records.")
    return df[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]]


# ==========================================================
# 2. Feature engineering (same as before)
# ==========================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()

    # --- Moving averages and exponential MAs ---
    for window in [5, 10, 20]:
        base[f"sma_{window}"] = base["Close"].rolling(window).mean()
        base[f"ema_{window}"] = base["Close"].ewm(span=window, adjust=False).mean()

    # --- RSI ---
    delta = base["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    base["rsi"] = 100 - (100 / (1 + rs))

    # --- Returns & volatility ---
    base["returns_1d"] = base["Close"].pct_change()
    base["returns_5d"] = base["Close"].pct_change(5)
    base["returns_10d"] = base["Close"].pct_change(10)
    base["volatility_5d"] = base["returns_1d"].rolling(5).std()
    base["volatility_20d"] = base["returns_1d"].rolling(20).std()

    # --- Price spreads ---
    base["high_low_pct"] = (base["High"] - base["Low"]) / base["Low"]
    base["volume_sma_5"] = base["Volume"].rolling(5).mean()
    base["volume_ratio"] = base["Volume"] / (base["volume_sma_5"] + 1e-9)

    # --- Lagged Close prices & returns ---
    for lag in [1, 2, 3, 5, 10]:
        base[f"Close_lag_{lag}"] = base["Close"].shift(lag)
        base[f"Close_ret_{lag}d"] = base["Close"].pct_change(lag)

    # --- Derived features ---
    base["Open_ret_1d"] = base["Open"].pct_change()
    base["High_ret_1d"] = base["High"].pct_change()
    base["Low_ret_1d"] = base["Low"].pct_change()
    base["Close_log"] = np.log(base["Close"] + 1e-9)

    for w in [3, 5, 10, 20]:
        base[f"Close_sma_{w}"] = base["Close"].rolling(w).mean()
        base[f"Close_std_{w}"] = base["Close"].rolling(w).std()
        base[f"Close_vol_{w}"] = base[f"Close_std_{w}"] / (base[f"Close_sma_{w}"] + 1e-9)

    # --- Time features ---
    base["date"] = pd.to_datetime(base["date"])
    base["dow"] = base["date"].dt.dayofweek
    base["month"] = base["date"].dt.month
    base["is_month_end"] = base["date"].dt.is_month_end.astype(int)

    # --- Momentum ---
    base["return_1d"] = base["Close"].pct_change()
    base["return_3d"] = base["Close"].pct_change(3)
    base["momentum_3d"] = base["Close"] / base["Close"].shift(3) - 1

    # --- Sentiment placeholders ---
    base["sentiment_score_mean"] = 0.0
    base["sentiment_score_std"] = 0.0
    base["sentiment_count"] = 0

    # Fill features safely
    feature_cols = [
        c for c in base.columns if c not in ["Open", "High", "Low", "Close", "Volume", "date", "ticker"]
    ]
    base[feature_cols] = base[feature_cols].bfill().ffill().fillna(0)

    return base


# ==========================================================
# 3. Full data refresh
# ==========================================================
def refresh_all():
    ensure_dir(STOCK_DATA_PATH.parent)
    print("Refreshing full stock dataset...")
    df = fetch_full_stock_data()
    df.to_csv(STOCK_DATA_PATH, index=False)
    print(f"Saved raw stock data → {STOCK_DATA_PATH}")

    print("Generating enhanced feature dataset...")
    df_feat = compute_features(df)
    df_feat.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Enhanced features saved → {FEATURES_DATA_PATH}")
    print(f"{len(df_feat)} rows | {len(df_feat.columns)} columns")
    print("Preview:\n", df_feat.tail())
    return df_feat


# ==========================================================
# 4. Compatibility alias (so Streamlit still works)
# ==========================================================
def update_features_enhanced():
    """
    Compatibility alias for dashboard and prediction scripts.
    Refreshes stock + feature datasets (same as refresh_all()).
    """
    print("[Alias] update_features_enhanced() → refresh_all()")
    return refresh_all()


# ==========================================================
# 5. Standalone run
# ==========================================================
if __name__ == "__main__":
    df = refresh_all()
    print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")
