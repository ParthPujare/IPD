"""
Prediction module for loading saved models and making predictions.
Supports LSTM and TFT models.
"""

import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, get_device

# Lazy imports for training modules (only load when models are actually used)
try:
    from src.training.train_lstm import LSTMModel
except ImportError:
    LSTMModel = None

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
except ImportError:
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None


FEATURES_DATA_PATH = get_project_root() / "data" / "features.csv"
SAVED_MODELS_DIR = get_project_root() / "models" / "saved_models"
CHECKPOINT_DIR = get_project_root() / "models" / "checkpoints"


def load_lstm_model():
    """
    Load trained LSTM model and required artifacts.
    
    Returns:
        tuple: (model, scaler, feature_cols, seq_len)
    """
    if LSTMModel is None:
        raise ImportError("LSTMModel not available. Please install pytorch-lightning.")
    
    model_path = SAVED_MODELS_DIR / "lstm.pt"
    scaler_path = SAVED_MODELS_DIR / "lstm_scaler.pkl"
    features_path = SAVED_MODELS_DIR / "lstm_features.pkl"
    checkpoint_path = list(CHECKPOINT_DIR.glob("lstm*.ckpt"))
    
    if not checkpoint_path:
        raise FileNotFoundError("LSTM checkpoint not found. Please train the model first.")
    
    checkpoint_path = checkpoint_path[0]
    
    # Load feature columns and scaler
    with open(features_path, "rb") as f:
        feature_cols = pickle.load(f)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load model from checkpoint
    device = get_device()
    model = LSTMModel.load_from_checkpoint(str(checkpoint_path))
    model.to(device)
    model.eval()
    
    # Infer seq_len from model or use default
    seq_len = 30  # Default, can be inferred from checkpoint
    
    return model, scaler, feature_cols, seq_len


def load_tft_model():
    """
    Load trained TFT model and required artifacts.
    
    Returns:
        tuple: (model, training_dataset)
    """
    if TemporalFusionTransformer is None:
        raise ImportError("TemporalFusionTransformer not available. Please install pytorch-forecasting.")
    
    model_path = SAVED_MODELS_DIR / "tft.pth"
    training_dataset_path = SAVED_MODELS_DIR / "tft_training_dataset.pkl"
    checkpoint_path = list(CHECKPOINT_DIR.glob("tft*.ckpt"))
    
    if not checkpoint_path:
        raise FileNotFoundError("TFT checkpoint not found. Please train the model first.")
    
    checkpoint_path = checkpoint_path[0]
    
    # Load training dataset
    with open(training_dataset_path, "rb") as f:
        training_dataset = pickle.load(f)
    
    # Load model from checkpoint
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
    device = get_device()
    # TFT models handle device internally, but ensure it's in eval mode
    model.eval()
    
    return model, training_dataset


def predict_next_day_lstm():
    """
    Predict next-day closing price using LSTM model.
    
    Returns:
        dict: Prediction results with 'predicted_price', 'confidence', and 'uncertainty'
    """
    try:
        # Load model
        model, scaler, feature_cols, seq_len = load_lstm_model()
        
        # Load features
        df = pd.read_csv(FEATURES_DATA_PATH)
        df = df.sort_values("date").reset_index(drop=True)
        
        # Get last sequence
        exclude_cols = ["date", "ticker", "sentiment_label_mode", "Close"]
        X = df[feature_cols].values
        
        # Take last seq_len rows
        if len(X) < seq_len:
            raise ValueError(f"Not enough data: need {seq_len} rows, have {len(X)}")
        
        sequence = X[-seq_len:]
        
        # Scale
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_scaled = scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(sequence.shape)
        
        # Predict
        device = get_device()
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(sequence_tensor)
            predicted_price = prediction.cpu().item()
        
        # Get last actual price for comparison
        last_price = df["Close"].iloc[-1]
        
        # Simple uncertainty estimate (can be improved with ensemble or uncertainty quantification)
        price_change_pct = abs((predicted_price - last_price) / last_price) * 100
        confidence = max(0, 100 - price_change_pct)  # Simple confidence metric
        
        return {
            "predicted_price": float(predicted_price),
            "last_price": float(last_price),
            "predicted_change_pct": float((predicted_price - last_price) / last_price * 100),
            "confidence": float(confidence),
            "uncertainty": float(price_change_pct)
        }
    
    except Exception as e:
        print(f"Error predicting with LSTM: {e}")
        raise


def predict_next_day_tft():
    """
    Predict next-day closing price using TFT model.
    
    Returns:
        dict: Prediction results with 'predicted_price', 'confidence', and 'uncertainty'
    """
    try:
        # Load model and dataset
        model, training_dataset = load_tft_model()
        
        # Load features
        df = pd.read_csv(FEATURES_DATA_PATH)
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        
        # Prepare data in TFT format
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days
        df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")
        
        # Create prediction dataset
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=True,
            stop_randomization=True
        )
        
        # Create dataloader
        prediction_dataloader = prediction_dataset.to_dataloader(
            train=False, batch_size=1, num_workers=0
        )
        
        # Predict
        predictions = model.predict(prediction_dataloader, return_y=True)
        
        # Extract prediction (last value)
        predicted_price = float(predictions.output.numpy()[-1, 0])
        
        # Get last actual price
        last_price = float(df["Close"].iloc[-1])
        
        # Confidence metric
        price_change_pct = abs((predicted_price - last_price) / last_price) * 100
        confidence = max(0, 100 - price_change_pct)
        
        return {
            "predicted_price": predicted_price,
            "last_price": last_price,
            "predicted_change_pct": float((predicted_price - last_price) / last_price * 100),
            "confidence": float(confidence),
            "uncertainty": float(price_change_pct)
        }
    
    except Exception as e:
        print(f"Error predicting with TFT: {e}")
        raise


def predict_next_day(model_name="LSTM"):
    """
    Predict next-day closing price using specified model.
    
    Args:
        model_name (str): Model name ('LSTM' or 'TFT')
    
    Returns:
        dict: Prediction results
    """
    if model_name.upper() == "LSTM":
        return predict_next_day_lstm()
    elif model_name.upper() == "TFT":
        return predict_next_day_tft()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'LSTM' or 'TFT'")


if __name__ == "__main__":
    # Test prediction
    print("Testing LSTM prediction...")
    try:
        result_lstm = predict_next_day("LSTM")
        print(f"LSTM Prediction: {result_lstm}")
    except Exception as e:
        print(f"LSTM prediction failed: {e}")
    
    print("\nTesting TFT prediction...")
    try:
        result_tft = predict_next_day("TFT")
        print(f"TFT Prediction: {result_tft}")
    except Exception as e:
        print(f"TFT prediction failed: {e}")

