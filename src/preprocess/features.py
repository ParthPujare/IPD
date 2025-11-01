"""
Feature engineering module for stock price prediction.
Computes technical indicators and merges with sentiment data.
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
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Ensure Close column exists
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    # Simple Moving Averages
    df["sma_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["sma_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["sma_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    
    # Exponential Moving Averages
    df["ema_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Returns
    df["returns_1d"] = df["Close"].pct_change(1)
    df["returns_5d"] = df["Close"].pct_change(5)
    df["returns_10d"] = df["Close"].pct_change(10)
    
    # Volatility (rolling standard deviation of returns)
    df["volatility_5d"] = df["returns_1d"].rolling(window=5, min_periods=1).std()
    df["volatility_20d"] = df["returns_1d"].rolling(window=20, min_periods=1).std()
    
    # Price position within high-low range
    df["high_low_pct"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-10)
    
    # Volume features
    df["volume_sma_5"] = df["Volume"].rolling(window=5, min_periods=1).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_5"] + 1e-10)
    
    # Lag features
    df["close_lag_1"] = df["Close"].shift(1)
    df["close_lag_5"] = df["Close"].shift(5)
    
    # Fill NaN values with forward fill and then backward fill
    df = df.ffill().bfill()
    
    return df


def aggregate_sentiment_by_date(news_df):
    """
    Aggregate sentiment scores by date.
    
    Args:
        news_df (pd.DataFrame): DataFrame with news data and sentiment scores
    
    Returns:
        pd.DataFrame: DataFrame with daily aggregated sentiment metrics
    """
    if news_df.empty or "sentiment_score" not in news_df.columns:
        return pd.DataFrame()
    
    df = news_df.copy()
    
    # Ensure published_date is datetime
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"]).dt.date
    
    # Aggregate by date
    daily_sentiment = df.groupby("published_date").agg({
        "sentiment_score": ["mean", "std", "count"],
        "sentiment_label": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "neutral"
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = [
        "date",
        "sentiment_score_mean",
        "sentiment_score_std",
        "sentiment_count",
        "sentiment_label_mode"
    ]
    
    # Fill NaN std with 0
    daily_sentiment["sentiment_score_std"] = daily_sentiment["sentiment_score_std"].fillna(0)
    
    return daily_sentiment


def build_features(stock_df=None, news_df=None):
    """
    Build features by merging stock data with technical indicators and sentiment.
    
    Args:
        stock_df (pd.DataFrame, optional): Stock price data. If None, loads from CSV.
        news_df (pd.DataFrame, optional): News data with sentiment. If None, loads from CSV.
    
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    # Load stock data if not provided
    if stock_df is None:
        if STOCK_DATA_PATH.exists():
            stock_df = pd.read_csv(STOCK_DATA_PATH)
            stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
        else:
            raise FileNotFoundError(f"Stock data not found at {STOCK_DATA_PATH}")
    
    # Load news data if not provided
    if news_df is None:
        if NEWS_DATA_PATH.exists():
            news_df = pd.read_csv(NEWS_DATA_PATH)
        else:
            print(f"Warning: News data not found at {NEWS_DATA_PATH}, proceeding without sentiment")
            news_df = pd.DataFrame()
    
    # Ensure stock_df date is datetime.date
    if isinstance(stock_df["date"].iloc[0], str):
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    elif isinstance(stock_df["date"].iloc[0], pd.Timestamp):
        stock_df["date"] = stock_df["date"].dt.date
    
    # Compute technical indicators
    print("Computing technical indicators...")
    features_df = compute_technical_indicators(stock_df)
    
    # Aggregate sentiment by date if news data exists
    if not news_df.empty and "sentiment_score" in news_df.columns:
        print("Aggregating sentiment by date...")
        daily_sentiment = aggregate_sentiment_by_date(news_df)
        
        if not daily_sentiment.empty:
            # Merge sentiment with stock data
            features_df = features_df.merge(
                daily_sentiment,
                on="date",
                how="left"
            )
            
            # Fill missing sentiment with neutral (0.0)
            features_df["sentiment_score_mean"] = features_df["sentiment_score_mean"].fillna(0.0)
            features_df["sentiment_score_std"] = features_df["sentiment_score_std"].fillna(0.0)
            features_df["sentiment_count"] = features_df["sentiment_count"].fillna(0)
            features_df["sentiment_label_mode"] = features_df["sentiment_label_mode"].fillna("neutral")
        else:
            print("Warning: No sentiment data to merge")
            # Add empty sentiment columns
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
    
    # Ensure date column is present and sorted
    features_df = features_df.sort_values("date").reset_index(drop=True)
    
    # Remove any remaining NaN values
    features_df = features_df.ffill().bfill()
    
    # Save features
    ensure_dir(FEATURES_DATA_PATH.parent)
    features_df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Features saved to {FEATURES_DATA_PATH}")
    print(f"Total features: {len(features_df.columns)} columns, {len(features_df)} rows")
    
    return features_df


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")
    
    try:
        features_df = build_features()
        print(f"\nFeature columns: {list(features_df.columns)}")
        print(f"\nSample data (last 5 rows):")
        print(features_df.tail())
        print(f"\nData shape: {features_df.shape}")
        print(f"\nDate range: {features_df['date'].min()} to {features_df['date'].max()}")
    except Exception as e:
        print(f"Error: {e}")

