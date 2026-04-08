"""
Multi-Stock Utility to track and record daily prediction values.
Stores: data/prediction_history.csv
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root

PREDICTION_HISTORY_FILE = get_project_root() / "data" / "prediction_history.csv"
STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"

def init_tracking_file():
    """Create the history CSV with a Ticker column if it doesn't exist."""
    if not PREDICTION_HISTORY_FILE.exists():
        df = pd.DataFrame(columns=[
            "Prediction_Date", 
            "Ticker",  # Added for Multi-Stock support
            "LSTM_Prediction", 
            "TFT_Prediction", 
            "Ensemble_Prediction", 
            "Actual_Close_Price"
        ])
        df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print(f"✅ Created prediction history file at {PREDICTION_HISTORY_FILE}")

def update_actual_prices():
    """Matches historical predictions with actual market outcomes from stock_data.csv."""
    if not PREDICTION_HISTORY_FILE.exists() or not STOCK_DATA_PATH.exists():
        return

    history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
    stock_df = pd.read_csv(STOCK_DATA_PATH)
    
    # Standardize dates
    history_df["Prediction_Date"] = pd.to_datetime(history_df["Prediction_Date"]).dt.date
    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    
    updated = False
    for idx, row in history_df.iterrows():
        # Only check rows where Actual_Close_Price is missing
        if pd.isna(row["Actual_Close_Price"]):
            ticker = row["Ticker"]
            pred_date = row["Prediction_Date"]
            
            # Find the first available trading day AFTER the prediction was made
            future_data = stock_df[(stock_df["ticker"] == ticker) & (stock_df["date"] > pred_date)]
            
            if not future_data.empty:
                # Get the 'Close' price of the next chronological trading day
                next_day_price = future_data.sort_values("date").iloc[0]["Close"]
                history_df.at[idx, "Actual_Close_Price"] = round(next_day_price, 2)
                updated = True

    if updated:
        history_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print("📈 Backfilled actual close prices in prediction history.")

def record_today_prediction(ticker, lstm_pred, tft_pred, ensemble_pred):
    """Stores predictions for a specific ticker."""
    init_tracking_file()
    
    # Load history
    history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
    history_df["Prediction_Date"] = pd.to_datetime(history_df["Prediction_Date"]).dt.date
    
    today = datetime.now().date()
    
    new_data = {
        "Prediction_Date": today,
        "Ticker": ticker,
        "LSTM_Prediction": round(lstm_pred, 2) if lstm_pred else None,
        "TFT_Prediction": round(tft_pred, 2) if tft_pred else None,
        "Ensemble_Prediction": round(ensemble_pred, 2) if ensemble_pred else None,
        "Actual_Close_Price": None
    }
    
    # Check if a record for this ticker on this date already exists
    mask = (history_df["Prediction_Date"] == today) & (history_df["Ticker"] == ticker)
    
    if not history_df.empty and mask.any():
        idx = history_df[mask].index[0]
        for col in ["LSTM_Prediction", "TFT_Prediction", "Ensemble_Prediction"]:
            history_df.at[idx, col] = new_data[col]
        print(f"🔄 Updated record for {ticker} on {today}.")
    else:
        new_row = pd.DataFrame([new_data])
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        print(f"📝 Added new record for {ticker} on {today}.")

    history_df.to_csv(PREDICTION_HISTORY_FILE, index=False)

if __name__ == "__main__":
    # Test script for HDFC Bank
    print("Testing Multi-Stock tracking...")
    # These would normally come from your dashboard/inference scripts
    record_today_prediction("HDFCBANK.NS", 768.27, 755.10, 762.00)
    update_actual_prices()