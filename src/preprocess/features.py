"""
Feature engineering module for stock price prediction.
Computes technical indicators PER TICKER and merges with sentiment data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir


STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"
NEWS_DATA_PATH = get_project_root() / "data" / "news_data.csv"
FEATURES_DATA_PATH = get_project_root() / "data" / "features.csv"


def compute_technical_indicators(df):
    """
    Compute technical indicators for stock price data.
    SAFE FOR MULTIPLE TICKERS: Processes each ticker separately so math doesn't overlap.
    """
    df = df.copy()
    
    # If ticker column is missing (fallback), create a dummy one
    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # We group by ticker so rolling averages don't bleed between different stocks
    grouped = df.groupby("ticker")

    # Simple Moving Averages
    df["sma_5"] = grouped["Close"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df["sma_10"] = grouped["Close"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df["sma_20"] = grouped["Close"].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    
    # Exponential Moving Averages
    df["ema_5"] = grouped["Close"].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df["ema_10"] = grouped["Close"].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df["ema_20"] = grouped["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    
    # RSI
    def calc_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
        
    df["rsi"] = grouped["Close"].transform(calc_rsi)
    
    # Returns
    df["returns_1d"] = grouped["Close"].transform(lambda x: x.pct_change(1))
    df["returns_5d"] = grouped["Close"].transform(lambda x: x.pct_change(5))
    df["returns_10d"] = grouped["Close"].transform(lambda x: x.pct_change(10))
    
    # Volatility
    df["volatility_5d"] = grouped["returns_1d"].transform(lambda x: x.rolling(window=5, min_periods=1).std())
    df["volatility_20d"] = grouped["returns_1d"].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    
    # Price position
    df["high_low_pct"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-10)
    
    # Volume features
    df["volume_sma_5"] = grouped["Volume"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_5"] + 1e-10)
    
    # Lag features
    df["close_lag_1"] = grouped["Close"].transform(lambda x: x.shift(1))
    df["close_lag_5"] = grouped["Close"].transform(lambda x: x.shift(5))
    
    # Clean up NaNs safely within each ticker group
    df = grouped.apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    return df


def aggregate_sentiment(news_df):
    """
    Aggregate sentiment scores by BOTH date and ticker.
    """
    if news_df.empty or "sentiment_score" not in news_df.columns:
        return pd.DataFrame()
    
    df = news_df.copy()
    
    # Ensure ticker exists
    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"
    
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"]).dt.date
    
    # Aggregate by date AND ticker
    daily_sentiment = df.groupby(["published_date", "ticker"]).agg({
        "sentiment_score": ["mean", "std", "count"],
        "sentiment_label": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "neutral"
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = [
        "date",
        "ticker",
        "sentiment_score_mean",
        "sentiment_score_std",
        "sentiment_count",
        "sentiment_label_mode"
    ]
    
    # Fill NaN std with 0
    daily_sentiment["sentiment_score_std"] = daily_sentiment["sentiment_score_std"].fillna(0)
    
    return daily_sentiment


def build_features(stock_df=None, news_df=None):
    if stock_df is None:
        if STOCK_DATA_PATH.exists():
            stock_df = pd.read_csv(STOCK_DATA_PATH)
            stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
        else:
            raise FileNotFoundError(f"Stock data not found at {STOCK_DATA_PATH}")
    
    if news_df is None:
        if NEWS_DATA_PATH.exists():
            news_df = pd.read_csv(NEWS_DATA_PATH)
        else:
            print(f"Warning: News data not found at {NEWS_DATA_PATH}")
            news_df = pd.DataFrame()
    
    if isinstance(stock_df["date"].iloc[0], str):
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    elif isinstance(stock_df["date"].iloc[0], pd.Timestamp):
        stock_df["date"] = stock_df["date"].dt.date
    
    print("Computing technical indicators for all tickers...")
    features_df = compute_technical_indicators(stock_df)
    
    if not news_df.empty and "sentiment_score" in news_df.columns:
        print("Aggregating sentiment by date and ticker...")
        daily_sentiment = aggregate_sentiment(news_df)
        
        if not daily_sentiment.empty:
            # Merge on BOTH date and ticker
            features_df = features_df.merge(
                daily_sentiment,
                on=["date", "ticker"],
                how="left"
            )
            
            features_df["sentiment_score_mean"] = features_df["sentiment_score_mean"].fillna(0.0)
            features_df["sentiment_score_std"] = features_df["sentiment_score_std"].fillna(0.0)
            features_df["sentiment_count"] = features_df["sentiment_count"].fillna(0)
            features_df["sentiment_label_mode"] = features_df["sentiment_label_mode"].fillna("neutral")
        else:
            print("Warning: No sentiment data to merge")
            features_df["sentiment_score_mean"] = 0.0
            features_df["sentiment_score_std"] = 0.0
            features_df["sentiment_count"] = 0
            features_df["sentiment_label_mode"] = "neutral"
    else:
        print("No sentiment data available, adding neutral sentiment columns")
        features_df["sentiment_score_mean"] = 0.0
        features_df["sentiment_score_std"] = 0.0
        features_df["sentiment_count"] = 0
        features_df["sentiment_label_mode"] = "neutral"
    
    features_df = features_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # We must group by ticker again when backfilling so NaNs don't grab data from other stocks
    features_df = features_df.groupby("ticker").apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    ensure_dir(FEATURES_DATA_PATH.parent)
    features_df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Features saved to {FEATURES_DATA_PATH}")
    print(f"Total features: {len(features_df.columns)} columns, {len(features_df)} rows")
    
    return features_df


if __name__ == "__main__":
    print("Testing feature engineering...")
    try:
        features_df = build_features()
        print(f"\nFeature columns: {list(features_df.columns)}")
        print(f"\nData shape: {features_df.shape}")
    except Exception as e:
        print(f"Error: {e}")