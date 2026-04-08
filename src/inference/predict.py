import pandas as pd
import numpy as np
import torch
import pickle
import sys
import os
from pathlib import Path
import warnings
import logging

# Silence library-specific warnings for a cleaner terminal
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- CRITICAL PATH FIX ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2] 
sys.path.insert(0, str(project_root))

from src.training.train_lstm import LSTMModel
from src.preprocess.shared_preprocessor import prepare_shared_data
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE # Required for output layer shape fix

# Strict Pathing
SAVED_MODELS_DIR = project_root / "models" / "saved_models"

def get_prediction_data(ticker):
    """Fetches the latest 30-day sequence for inference."""
    df_processed, feature_cols = prepare_shared_data(is_training=False)
    df = df_processed[df_processed['ticker'] == ticker].sort_values('date')
    if len(df) < 30: 
        return None, None, None
    last_price = df['Close'].iloc[-1]
    sequence = df[feature_cols].values[-30:]
    return sequence, last_price, feature_cols

def predict_lstm(ticker):
    """Inference logic for ticker-specific LSTM models."""
    try:
        sequence, last_price, feature_cols = get_prediction_data(ticker)
        if sequence is None: return None
        
        model_path = SAVED_MODELS_DIR / f"lstm_{ticker}.pt"
        scaler_path = SAVED_MODELS_DIR / "shared_target_scaler.pkl"
        
        with open(scaler_path, "rb") as f:
            target_scaler = pickle.load(f)

        model = LSTMModel(input_size=len(feature_cols))
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()

        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(seq_tensor).numpy()
        
        price = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        return {"price": price, "last": last_price}
    except Exception as e:
        print(f"❌ LSTM Error for {ticker}: {e}")
        return None

def predict_tft(ticker):
    """Inference logic for the Global TFT model with explicit output shape fix."""
    try:
        dataset_path = SAVED_MODELS_DIR / "tft_training_dataset.pkl"
        model_weights = SAVED_MODELS_DIR / "tft_global.pt"
        
        with open(dataset_path, "rb") as f:
            training_ds = pickle.load(f)
        
        # --- FIXED: Explicitly set architecture AND loss to match 1-value output ---
        model = TemporalFusionTransformer.from_dataset(
            training_ds,
            hidden_size=64,
            attention_head_size=2,
            hidden_continuous_size=64,
            output_size=1,   # Fixed: Prevents the [7] vs [1] shape mismatch
            loss=MAE()       # Fixed: Matches the training loss function
        )
        
        model.load_state_dict(torch.load(model_weights, map_location='cpu', weights_only=True))
        model.eval()

        df_all, _ = prepare_shared_data(is_training=False)
        df_all["time_idx"] = df_all.groupby("ticker").cumcount()
        df_all["ticker"] = df_all["ticker"].astype(str)
        
        # Generate prediction
        prediction = model.predict(df_all[df_all.ticker == ticker], mode="prediction")
        
        scaler_path = SAVED_MODELS_DIR / "shared_target_scaler.pkl"
        with open(scaler_path, "rb") as f:
            target_scaler = pickle.load(f)
            
        price = float(target_scaler.inverse_transform(prediction.numpy().reshape(-1, 1))[0][0])
        last_price = df_all[df_all.ticker == ticker]['Close'].iloc[-1]
        
        return {"price": price, "last": last_price}
    except Exception as e:
        print(f"❌ TFT Error for {ticker}: {e}")
        return None

def predict_next_day(model_name="LSTM", ticker="ADANIGREEN.NS"):
    """Entry point for Dashboard and Ensemble logic."""
    res = None
    if model_name == "LSTM":
        res = predict_lstm(ticker)
    elif model_name == "TFT":
        res = predict_tft(ticker)
    elif model_name == "Ensemble":
        l_res = predict_lstm(ticker)
        t_res = predict_tft(ticker)
        if l_res and t_res:
            blend = (l_res["price"] * 0.6) + (t_res["price"] * 0.4)
            res = {"price": blend, "last": l_res["last"]}
        else:
            res = l_res or t_res
    else: 
        return None

    if res:
        return {
            "predicted_price": res["price"],
            "last_price": res["last"],
            "predicted_change_pct": ((res["price"] - res["last"]) / res["last"]) * 100
        }
    return None