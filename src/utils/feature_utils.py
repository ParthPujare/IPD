# src/utils/feature_utils.py
from typing import List
import pandas as pd
import numpy as np

def add_basic_technical_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Adds low-risk rolling, lag and time features to dataframe. Returns a new dataframe copy.
    Non-destructive: does not mutate original DataFrame passed in.
    """
    df = df.copy()
    # Ensure date is datetime if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Basic safe transforms
    #  - log price (handle zeros)
    #  - daily returns
    df[f"{price_col}_log"] = np.log(df[price_col].replace(0, np.nan)).fillna(method="ffill")
    df[f"{price_col}_ret_1d"] = df[price_col].pct_change().fillna(0)

    # Rolling means and std (common windows)
    windows = [3, 5, 10, 20]
    for w in windows:
        df[f"{price_col}_sma_{w}"] = df[price_col].rolling(w, min_periods=1).mean()
        df[f"{price_col}_std_{w}"] = df[price_col].rolling(w, min_periods=1).std().fillna(0)
        # volatility normalized by sma to avoid scale issues; safe division
        sma = df[f"{price_col}_sma_{w}"].replace(0, np.nan)
        df[f"{price_col}_vol_{w}"] = (df[f"{price_col}_std_{w}"] / sma).replace([np.inf, -np.inf], 0).fillna(0)

    # Lags and multi-day returns
    lag_cols = [1, 2, 3, 5, 10]
    for l in lag_cols:
        df[f"{price_col}_lag_{l}"] = df[price_col].shift(l)
        df[f"{price_col}_ret_{l}d"] = df[price_col].pct_change(l).fillna(0)

    # Fill or backfill lagged NaNs conservatively
    lag_fill_cols = [c for c in df.columns if f"{price_col}_lag" in c]
    for c in lag_fill_cols:
        df[c] = df[c].fillna(method="bfill").fillna(method="ffill")

    # Volume based features if available
    if "Volume" in df.columns:
        df["volume_sma_5"] = df["Volume"].rolling(5, min_periods=1).mean()
        vol_sma_20 = df["Volume"].rolling(20, min_periods=1).mean().replace(0, np.nan)
        df["volume_ratio"] = (df["Volume"] / vol_sma_20).replace([np.inf, -np.inf], 0).fillna(1)

    # Price OHLC features presence (safe adds)
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[f"{col}_ret_1d"] = df[col].pct_change().fillna(0)

    # Time features (weekday, month, month_end)
    if "date" in df.columns:
        df["dow"] = df["date"].dt.dayofweek.astype("int8")
        df["month"] = df["date"].dt.month.astype("int8")
        df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")

    # Replace remaining infinite values and fill NA safely
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(0, inplace=True)

    return df

def select_feature_cols(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Return a list of feature columns for modeling by excluding the given list.
    Default excludes common non-feature columns.
    """
    exclude = exclude or ["date", "ticker"]
    return [c for c in df.columns if c not in exclude]
