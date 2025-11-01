"""
Fetch and store historical stock prices for ADANIGREEN.NS using yfinance.
Handles data fetching, deduplication, and CSV storage.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir


TICKER = "ADANIGREEN.NS"
STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"


def fetch_stock_data(period="5y"):
    """
    Fetch historical stock data from yfinance.
    
    Args:
        period (str): Period to fetch. Options: '5y', 'max', etc.
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        ticker = yf.Ticker(TICKER)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data fetched for {TICKER}")
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "date"}, inplace=True)
        
        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        # Add ticker column
        df["ticker"] = TICKER
        
        # Reorder columns
        cols = ["date", "ticker", "Open", "High", "Low", "Close", "Volume"]
        df = df[cols]
        
        print(f"Fetched {len(df)} records for {TICKER}")
        return df
    
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        raise


def update_stock_data():
    """
    Update stock data CSV by fetching latest data and deduplicating.
    Appends new data to existing file or creates new file if it doesn't exist.
    """
    ensure_dir(STOCK_DATA_PATH.parent)
    
    # Fetch latest data
    print(f"Fetching stock data for {TICKER}...")
    new_df = fetch_stock_data(period="5y")
    
    # Load existing data if it exists
    if STOCK_DATA_PATH.exists():
        try:
            existing_df = pd.read_csv(STOCK_DATA_PATH)
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
            
            # Merge and deduplicate by date
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
            combined_df = combined_df.sort_values("date").reset_index(drop=True)
            
            print(f"Updated stock data: {len(existing_df)} -> {len(combined_df)} records")
        except Exception as e:
            print(f"Error reading existing data, using new data only: {e}")
            combined_df = new_df
    else:
        combined_df = new_df
        print(f"Created new stock data file with {len(combined_df)} records")
    
    # Save to CSV
    combined_df.to_csv(STOCK_DATA_PATH, index=False)
    print(f"Stock data saved to {STOCK_DATA_PATH}")
    
    return combined_df


if __name__ == "__main__":
    # Test the fetch functionality
    print("Testing stock data fetch...")
    df = update_stock_data()
    print(f"\nSample data (last 5 rows):")
    print(df.tail())
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

