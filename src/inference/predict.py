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
    from pytorch_forecasting.metrics import QuantileLoss  # ✅ needed for proper TFT loss
except ImportError:
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    QuantileLoss = None


FEATURES_DATA_PATH = get_project_root() / "data" / "features.csv"
SAVED_MODELS_DIR = get_project_root() / "models" / "saved_models"
CHECKPOINT_DIR = get_project_root() / "models" / "checkpoints"


# ✅ Fix for PyTorch ≥ 2.6 (safe unpickling)
def safe_torch_load(path):
    """Safely load serialized PyTorch Forecasting objects."""
    from torch.serialization import add_safe_globals
    from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

    add_safe_globals([TimeSeriesDataSet])
    return torch.load(path, map_location="cpu", weights_only=False)


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

    if not model_path.exists():
        raise FileNotFoundError(f"TFT model not found at {model_path}. Please train it first.")
    if not training_dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {training_dataset_path}. Please train the model first.")

    #  Safe load for PyTorch >= 2.6
    from torch.serialization import add_safe_globals
    from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE  #  Lightning metric version

    add_safe_globals([TimeSeriesDataSet])

    with open(training_dataset_path, "rb") as f:
        training_dataset = pickle.load(f)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Rebuild model exactly as trained
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=1,
        loss=MAE(),  #  Correct Lightning Metric
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    print("TFT model loaded safely using MAE loss and manual checkpoint restore.")
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
        model, training_dataset = load_tft_model()

        # Load and prepare data
        df = pd.read_csv(FEATURES_DATA_PATH)
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days
        df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=True,
            stop_randomization=True
        )

        prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        # ✅ Wrap predict call in debug
        print("\n🚀 Running TFT prediction...")
        preds = model.predict(prediction_dataloader, return_y=False)

        print("\n========== TFT Prediction Debug Info ==========")
        print(f"Type: {type(preds)}")
        if isinstance(preds, torch.Tensor):
            print(f"Tensor shape: {preds.shape}")
            print(f"Tensor sample: {preds.flatten()[:10]}")
        elif isinstance(preds, (list, tuple)):
            print(f"List/Tuple length: {len(preds)}")
            for i, item in enumerate(preds):
                if isinstance(item, torch.Tensor):
                    print(f"  -> Item[{i}] Tensor shape: {item.shape}")
            if len(preds) > 0:
                print("Sample (first element):", preds[0])
        elif isinstance(preds, np.ndarray):
            print(f"Numpy shape: {preds.shape}")
            print(f"Numpy sample: {preds.flatten()[:10]}")
        else:
            print("Unknown type:", type(preds))
        print("===============================================")

        # ✅ Safely flatten output to handle shape errors
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(preds, (list, tuple)):
            preds = np.array(preds)
        preds_flat = preds.reshape(-1)
        predicted_price = float(preds_flat[-1])

        last_price = float(df["Close"].iloc[-1])
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
        import traceback
        traceback.print_exc()
        raise


def predict_next_day(model_name="LSTM"):
    """
    Predict next-day closing price using specified model.
    """
    if model_name.upper() == "LSTM":
        return predict_next_day_lstm()
    elif model_name.upper() == "TFT":
        return predict_next_day_tft()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'LSTM' or 'TFT'")


if __name__ == "__main__":
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
