"""
Utility to track and record daily prediction values for models.
Stores date, next day's actual price (updated later), and predicted prices from models.
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root

PREDICTION_HISTORY_FILE = get_project_root() / "data" / "prediction_history.csv"
STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"

def init_tracking_file():
    """Create the history CSV if it doesn't exist."""
    if not PREDICTION_HISTORY_FILE.exists():
        df = pd.DataFrame(columns=[
            "Prediction_Date", 
            "LSTM_Prediction", 
            "TFT_Prediction", 
            "Ensemble_Prediction", 
            "Actual_Close_Price"
        ])
        df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print(f"Created prediction history file at {PREDICTION_HISTORY_FILE}")

def update_actual_prices():
    """
    Check historical predictions and update the 'Actual_Close_Price'
    column based on the actual stock data that has since been recorded.
    """
    if not PREDICTION_HISTORY_FILE.exists() or not STOCK_DATA_PATH.exists():
        return

    history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
    stock_df = pd.read_csv(STOCK_DATA_PATH)
    
    # Ensure datetime format for matching
    history_df["Prediction_Date"] = pd.to_datetime(history_df["Prediction_Date"]).dt.date
    if "date" in stock_df.columns:
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    elif "Date" in stock_df.columns:
        stock_df["date"] = pd.to_datetime(stock_df["Date"]).dt.date
    else:
        return
        
    stock_dict = dict(zip(stock_df["date"], stock_df["Close"]))

    # Look for matching dates and update missing actual prices
    updated = False
    for idx, row in history_df.iterrows():
        pred_date = row["Prediction_Date"]
        # The actual closing price we want to predict is the NEXT trading day
        # Look for the next available date in the stock_dict
        if pd.isna(row["Actual_Close_Price"]):
            # Find the actual close price for the trading day AFTER the prediction date
            future_dates = [d for d in stock_dict.keys() if d > pred_date]
            if future_dates:
                next_trading_day = min(future_dates)
                history_df.at[idx, "Actual_Close_Price"] = stock_dict[next_trading_day]
                updated = True

    if updated:
        history_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print("Updated actual close prices in prediction history.")

def record_today_prediction(lstm_pred, tft_pred, ensemble_pred):
    """
    Store today's predictions in the CSV file. Updates existing row if 
    recorded today, else appends a new row.
    """
    init_tracking_file()
    update_actual_prices()
    
    history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
    history_df["Prediction_Date"] = pd.to_datetime(history_df["Prediction_Date"]).dt.date
    
    today = datetime.now().date()
    
    new_data = {
        "Prediction_Date": today,
        "LSTM_Prediction": f"{lstm_pred:.2f}" if lstm_pred else None,
        "TFT_Prediction": f"{tft_pred:.2f}" if tft_pred else None,
        "Ensemble_Prediction": f"{ensemble_pred:.2f}" if ensemble_pred else None,
        "Actual_Close_Price": None  # To be filled next trading day
    }
    
    # Check if we already recorded a prediction today
    if not history_df.empty and today in history_df["Prediction_Date"].values:
        idx = history_df[history_df["Prediction_Date"] == today].index[0]
        history_df.loc[idx, ["LSTM_Prediction", "TFT_Prediction", "Ensemble_Prediction"]] = [
            new_data["LSTM_Prediction"], new_data["TFT_Prediction"], new_data["Ensemble_Prediction"]
        ]
        print(f"Updated today's prediction record.")
    else:
        # Avoid appending dicts natively in newer pandas, use pd.concat
        new_row_df = pd.DataFrame([new_data])
        history_df = pd.concat([history_df, new_row_df], ignore_index=True)
        print(f"Added new prediction record for today.")

    history_df.to_csv(PREDICTION_HISTORY_FILE, index=False)

if __name__ == "__main__":
    from src.inference.predict import predict_next_day
    from src.inference.ensemble import ensemble_predict
    
    print("Testing tracking script...")
    lstm_res = predict_next_day("LSTM")
    tft_res = predict_next_day("TFT")
    ens_res = ensemble_predict()
    
    record_today_prediction(
        lstm_res["predicted_price"], 
        tft_res["predicted_price"], 
        ens_res["predicted_price"]
    )
    print("Test complete.")
