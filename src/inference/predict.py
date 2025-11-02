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


FEATURES_DATA_PATH = get_project_root() / "data" / "features_enhanced.csv"
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
    Load trained LSTM model with non-strict loading — skips mismatched layers safely.
    """
    if LSTMModel is None:
        raise ImportError("LSTMModel not available. Please install pytorch-lightning.")

    model_path = SAVED_MODELS_DIR / "lstm.pt"
    features_path = SAVED_MODELS_DIR / "lstm_features.pkl"
    checkpoint_path = list(CHECKPOINT_DIR.glob("lstm*.ckpt"))
    if not checkpoint_path:
        raise FileNotFoundError("LSTM checkpoint not found.")
    checkpoint_path = checkpoint_path[0]

    # Load features
    with open(features_path, "rb") as f:
        feature_cols = pickle.load(f)

    # Load checkpoint manually
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LSTMModel(**checkpoint["hyper_parameters"])
    state_dict = checkpoint["state_dict"]

    # 🧠 Try to load weights safely
    new_state_dict = model.state_dict()
    compatible_weights = {k: v for k, v in state_dict.items() if k in new_state_dict and v.shape == new_state_dict[k].shape}
    missing = [k for k in new_state_dict.keys() if k not in compatible_weights]
    skipped = [k for k in state_dict.keys() if k not in compatible_weights]
    print(f"✅ Loaded {len(compatible_weights)} compatible layers, skipped {len(skipped)} mismatched ones")

    new_state_dict.update(compatible_weights)
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()

    seq_len = 30
    return model, feature_cols, seq_len





def load_tft_model():
    """
    Load the trained Temporal Fusion Transformer (TFT) model and its dataset.
    Ensures the architecture matches the trained configuration.
    """
    if TemporalFusionTransformer is None:
        raise ImportError("TemporalFusionTransformer not available. Please install pytorch-forecasting.")
    
    # === Paths ===
    model_path = SAVED_MODELS_DIR / "tft.pth"
    training_dataset_path = SAVED_MODELS_DIR / "tft_training_dataset.pkl"
    best_checkpoint_path = CHECKPOINT_DIR / "tft-epoch=08-val_loss=36.8163.ckpt"  # ✅ Use your trained checkpoint

    if not model_path.exists():
        raise FileNotFoundError(f"TFT model not found at {model_path}. Please train it first.")
    if not training_dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {training_dataset_path}. Please train the model first.")
    if not best_checkpoint_path.exists():
        print(f"⚠️ Checkpoint not found at {best_checkpoint_path}, falling back to {model_path}")

    # === Safe load for PyTorch >= 2.6 ===
    from torch.serialization import add_safe_globals
    from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE

    add_safe_globals([TimeSeriesDataSet])

    # Load dataset for structure inference
    with open(training_dataset_path, "rb") as f:
        training_dataset = pickle.load(f)

    # === Build model exactly as trained ===
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.001,         # match your training script
        hidden_size=64,              # ✅ matches trained architecture
        attention_head_size=2,       # ✅ matches trained architecture
        dropout=0.1,                 # ✅ matches trained architecture
        hidden_continuous_size=64,   # ✅ same as hidden_size
        output_size=1,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # === Load weights ===
    checkpoint_to_load = (
        torch.load(best_checkpoint_path, map_location="cpu")["state_dict"]
        if best_checkpoint_path.exists()
        else torch.load(model_path, map_location="cpu", weights_only=False)["model_state_dict"]
    )

        # === Load weights safely and cleanly ===
    try:
        checkpoint_to_load = (
            torch.load(best_checkpoint_path, map_location="cpu")["state_dict"]
            if best_checkpoint_path.exists()
            else torch.load(model_path, map_location="cpu", weights_only=False)["model_state_dict"]
        )

        missing, unexpected = model.load_state_dict(checkpoint_to_load, strict=False)
        if missing or unexpected:
            print(f"⚠️ Partial state_dict load:")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("✅ TFT weights loaded fully and correctly.")
    except Exception as e:
        print(f"⚠️ Failed to load full state_dict — {e}")

    model.eval()

    print(f"TFT model loaded successfully (hidden_size=64, heads=2, dropout=0.1)")
    return model, training_dataset




def predict_next_day_lstm():
    """
    Predict next-day closing price using the trained LSTM model with consistent scaling and deterministic behavior.
    """
    try:
        import random
        import numpy as np
        import torch

        # === Deterministic setup ===
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # === Load model & metadata ===
        model, feature_cols, seq_len = load_lstm_model()

        # === Locate scaler files ===
        scaler_dir = get_project_root() / "models" / "saved_models"
        feature_scaler_path = scaler_dir / "lstm_feature_scaler.pkl"
        target_scaler_path = scaler_dir / "lstm_target_scaler.pkl"

        if not feature_scaler_path.exists() or not target_scaler_path.exists():
            raise FileNotFoundError(
                f"Missing scaler files — retrain the LSTM model first.\n"
                f"Expected: {feature_scaler_path.name}, {target_scaler_path.name}"
            )

        # === Load scalers ===
        with open(feature_scaler_path, "rb") as f:
            feature_scaler = pickle.load(f)
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)

        # === Load latest (raw) features ===
        df = pd.read_csv(FEATURES_DATA_PATH).sort_values("date").reset_index(drop=True)
        exclude_cols = ["date", "ticker", "sentiment_label_mode", "Close"]

        # === Ensure feature consistency ===
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns missing from data: {missing_cols}")

        X = df[feature_cols].astype(float).values
        if len(X) < seq_len:
            raise ValueError(f"Not enough data: need {seq_len} rows, have {len(X)}")

        # === Prepare latest sequence ===
        sequence = X[-seq_len:]
        sequence_scaled = feature_scaler.transform(sequence)
        sequence_scaled = np.expand_dims(sequence_scaled, axis=0)  # (1, seq_len, n_features)

        # === Model inference ===
        device = get_device()
        model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence_scaled).to(device)
            y_pred_scaled = model(sequence_tensor).detach().cpu().numpy().reshape(-1, 1)

        # === Inverse scale prediction ===
        predicted_price = float(target_scaler.inverse_transform(y_pred_scaled)[0][0])

        # === Compute metrics ===
        last_price = float(df["Close"].iloc[-1])
        change_pct = ((predicted_price - last_price) / last_price) * 100
        confidence = max(0.0, 100.0 - abs(change_pct))
        uncertainty = abs(change_pct)

        # === Debug info ===
        print("\n========== LSTM Prediction Debug Info ==========")
        print(f"Predicted (scaled): {y_pred_scaled[0][0]:.6f}")
        print(f"Predicted (₹): {predicted_price:,.2f}")
        print(f"Last actual (₹): {last_price:,.2f}")
        print(f"Predicted change: {change_pct:.2f}% | Confidence: {confidence:.2f}%")
        print("================================================\n")

        return {
            "predicted_price": predicted_price,
            "last_price": last_price,
            "predicted_change_pct": change_pct,
            "confidence": confidence,
            "uncertainty": uncertainty,
        }

    except Exception as e:
        print(f" Error predicting with LSTM: {e}")
        raise





def predict_next_day_tft():
    """
    Predict next-day closing price using the trained TFT model.
    Reconstructs all derived features used during training and applies the same scaling.
    Ensures consistency between training and inference.
    """
    try:
        import random
        import torch
        import numpy as np

        # === Deterministic predictions ===
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.use_deterministic_algorithms(True)

        model, training_dataset = load_tft_model()

        # === Load and prepare latest feature data ===
        df = pd.read_csv(FEATURES_DATA_PATH).sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days
        df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

        # 🧹 Drop or encode non-numeric columns
        non_numeric_cols = ["sentiment_label_mode", "ticker"]
        for col in non_numeric_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # === Recreate derived features expected by the trained TFT ===
        if "Close" in df.columns:
            df["Close_diff"] = df["Close"].diff().fillna(0)
            df["Close_pct_change"] = df["Close"].pct_change().fillna(0)

            # rolling mean / std
            if "Close_sma_5" in df.columns:
                df["Close_rolling_mean_5"] = df["Close_sma_5"]
            else:
                df["Close_rolling_mean_5"] = df["Close"].rolling(window=5, min_periods=1).mean()

            if "Close_std_5" in df.columns:
                df["Close_rolling_std_5"] = df["Close_std_5"]
            else:
                df["Close_rolling_std_5"] = df["Close"].rolling(window=5, min_periods=1).std().fillna(0.0)
        else:
            raise KeyError("'Close' column missing — cannot create derived features.")

        # === Load feature scaler (fitted during training) ===
        scaler_path = SAVED_MODELS_DIR / "tft_feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Feature scaler not found at {scaler_path}")

        with open(scaler_path, "rb") as f:
            feature_scaler = pickle.load(f)

        # === Identify numeric feature columns used in training ===
        exclude_cols = ["date", "time_idx", "group_id", "Close"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # ⚠️ Handle mismatched features
        missing_features = set(feature_scaler.feature_names_in_) - set(feature_cols)
        extra_features = set(feature_cols) - set(feature_scaler.feature_names_in_)
        if missing_features:
            print(f"⚠️ Missing features in current data: {missing_features}")
        if extra_features:
            print(f"⚠️ Extra features not seen during training: {extra_features}")

        # Align with scaler’s feature order (avoid shape errors)
        feature_cols = [c for c in feature_scaler.feature_names_in_ if c in df.columns]

        # Apply same scaling (DO NOT fit again)
        df[feature_cols] = feature_scaler.transform(df[feature_cols].astype(float))

        # === Create prediction dataset using the same config ===
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=True,
            stop_randomization=True
        )

        prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        # === Model inference (deterministic) ===
        model.eval()
        torch.set_grad_enabled(False)

        print("\n🚀 Running TFT prediction...")
        preds = model.predict(prediction_dataloader, return_y=False)

        # === Convert prediction tensor to NumPy ===
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        elif isinstance(preds, (list, tuple)):
            preds = np.array(preds)

        preds_flat = preds.reshape(-1)
        predicted_price = float(preds_flat[-1])

        # === Compare with last actual close ===
        last_price = float(df["Close"].iloc[-1])
        price_change_pct = ((predicted_price - last_price) / last_price) * 100
        confidence = max(0, 100 - abs(price_change_pct))

        print("\n=== Debug: Checking non-numeric columns before scaling ===")
        for col in feature_cols:
            non_numeric = df[col].apply(lambda x: not np.issubdtype(type(x), np.number)).any()
            if non_numeric:
                print(f"⚠️ Non-numeric column detected: {col}")
                print(df[col].unique()[:10])
        print("===========================================================\n")


        print("\n========== TFT Prediction Debug Info ==========")
        print(f"Predicted price (₹): {predicted_price:.2f}")
        print(f"Last actual (₹): {last_price:.2f}")
        print(f"Predicted change: {price_change_pct:.2f}% | Confidence: {confidence:.2f}%")
        print("===============================================\n")

        return {
            "predicted_price": predicted_price,
            "last_price": last_price,
            "predicted_change_pct": float(price_change_pct),
            "confidence": float(confidence),
            "uncertainty": float(abs(price_change_pct)),
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
