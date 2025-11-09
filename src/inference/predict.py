"""
Prediction module for loading saved models and making predictions.
Supports LSTM and TFT models with unified shared scaling.
This version is resilient to column-name casing/alias mismatches and will
attempt to map known feature aliases before raising errors.
"""

import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import sys
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, get_device

# Lazy imports
try:
    from src.training.train_lstm import LSTMModel
except ImportError:
    LSTMModel = None

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE
except ImportError:
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    MAE = None

# === Paths ===
FEATURES_DATA_PATH = get_project_root() / "data" / "features_enhanced.csv"
SAVED_MODELS_DIR = get_project_root() / "models" / "saved_models"
CHECKPOINT_DIR = get_project_root() / "models" / "checkpoints"


# === Utilities ===
def _col_exists_anycase(df, col):
    """Return actual column name in df that matches col case-insensitively, or None."""
    if col in df.columns:
        return col
    lower = col.lower()
    for c in df.columns:
        if c.lower() == lower:
            return c
    return None

def align_features_with_df(df, required_features):
    """
    Ensure df contains all required_features. For each required feature:
      - if exact name exists, keep
      - else if lowercase/uppercase variant exists, create required name as alias to that column
      - else try other common aliases (e.g. Close_sma_5 <-> Close_sma5 etc.)
    Returns new df (copy), list of missing features (if any).
    """
    df = df.copy()
    created = []
    missing = []
    for feat in required_features:
        found = _col_exists_anycase(df, feat)
        if found:
            # if case differs, create canonical column name with values from found
            if found != feat:
                df[feat] = df[found]
                created.append((feat, found))
            continue

        # try some alias patterns (common)
        aliases = [
            feat.lower(),
            feat.upper(),
            feat.replace("_", ""),
            feat.replace("_", "").lower(),
            feat.replace("_", "").upper()
        ]
        found_alias = None
        for a in aliases:
            for c in df.columns:
                if c.lower() == a.lower():
                    found_alias = c
                    break
            if found_alias:
                break

        if found_alias:
            df[feat] = df[found_alias]
            created.append((feat, found_alias))
            continue

        # not found
        missing.append(feat)

    return df, created, missing


def safe_read_features():
    """Read features CSV and coerce column names to strings; ensure date parsing."""
    if not FEATURES_DATA_PATH.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_DATA_PATH}")
    df = pd.read_csv(FEATURES_DATA_PATH)
    # ensure columns are strings
    df.columns = [str(c) for c in df.columns]
    # try parse date
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass
    elif "Date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["Date"])
            df.drop(columns=["Date"], inplace=True)
        except Exception:
            pass
    return df


# === Load LSTM ===
def load_lstm_model():
    if LSTMModel is None:
        raise ImportError("LSTMModel not available. Please install pytorch-lightning.")

    checkpoint_path = list(CHECKPOINT_DIR.glob("lstm*.ckpt"))
    if not checkpoint_path:
        raise FileNotFoundError("LSTM checkpoint not found.")
    checkpoint_path = checkpoint_path[0]

    with open(SAVED_MODELS_DIR / "shared_features.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Some checkpoints may store hyper_parameters under different keys; try both
    hyper = checkpoint.get("hyper_parameters", checkpoint.get("hyper_params", {}))
    # If hyper doesn't contain input_size, we'll construct model with input_size=len(feature_cols)
    if "input_size" in hyper:
        model = LSTMModel(**hyper)
    else:
        # fallback: construct model with sensible defaults
        model = LSTMModel(input_size=len(feature_cols),
                          hidden_size=hyper.get("hidden_size", 128),
                          num_layers=hyper.get("num_layers", 2),
                          dropout=hyper.get("dropout", 0.2),
                          lr=hyper.get("lr", 1e-3))
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Partial weight loading
    new_state_dict = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in new_state_dict and v.shape == new_state_dict[k].shape}
    new_state_dict.update(compatible)
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    seq_len = 30
    print(f"LSTM model loaded with {len(feature_cols)} shared features.")
    return model, feature_cols, seq_len


def load_tft_model():
    """
    Load the trained TFT model and its dataset.
    Preferred: load from a Lightning checkpoint (.ckpt) using load_from_checkpoint()
    Fallback: load weights from tft.pth into a model constructed from the training dataset.
    """
    if TemporalFusionTransformer is None:
        raise ImportError("TemporalFusionTransformer not available. Please install pytorch-forecasting.")

    model_path = SAVED_MODELS_DIR / "tft.pth"
    training_dataset_path = SAVED_MODELS_DIR / "tft_training_dataset.pkl"

    # prefer a full checkpoint if present (Lightning .ckpt in CHECKPOINT_DIR)
    ckpt_candidates = list(CHECKPOINT_DIR.glob("tft*.ckpt")) + list(CHECKPOINT_DIR.glob("*tft*.ckpt"))
    ckpt_candidates = sorted(ckpt_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_path = ckpt_candidates[0] if ckpt_candidates else None

    if not training_dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {training_dataset_path}. Please train the model first.")

    # load training dataset (structure is needed)
    with open(training_dataset_path, "rb") as f:
        training_dataset = pickle.load(f)

    device = "cpu"

    # 1) If we have a Lightning checkpoint, try loading full model from it (restores architecture)
    if checkpoint_path and checkpoint_path.exists():
        try:
            print(f"Attempting to load TFT from checkpoint: {checkpoint_path}")
            # pytorch-forecasting models are LightningModules, they support load_from_checkpoint
            model = TemporalFusionTransformer.load_from_checkpoint(
                str(checkpoint_path),
                map_location=device,
            )
            model.eval()
            print(" TFT loaded from .ckpt (architecture + weights restored).")
            return model, training_dataset
        except Exception as e:
            print(f" Loading from checkpoint failed: {e}")
            print("Falling back to loading weights/state_dict from tft.pth ...")

    # 2) Fallback: if tft.pth exists, try to construct model from training_dataset and load saved state dict.
    if model_path.exists():
        # construct model using training_dataset to ensure consistent encoders / known features
        print("Constructing TFT model from training_dataset configuration (fallback).")
        try:
            model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=0.001,
                hidden_size=64,
                attention_head_size=2,
                dropout=0.1,
                hidden_continuous_size=64,
                output_size=1,
                loss=MAE() if MAE is not None else None,
                log_interval=10,
                reduce_on_plateau_patience=4,
            )
        except Exception as e:
            # last-resort: instantiate with defaults (still may fail if architecture mismatches)
            print(f"Warning: from_dataset construction failed: {e}. Attempting default constructor.")
            model = TemporalFusionTransformer(
                hidden_size=64,
                attention_head_size=2,
                dropout=0.1,
                hidden_continuous_size=64,
                output_size=1,
                loss=MAE() if MAE is not None else None,
            )

        # load tft.pth contents
        weights = torch.load(model_path, map_location=device)
        state_dict = weights.get("model_state_dict") if isinstance(weights, dict) and "model_state_dict" in weights else weights

        # try load with strict=False so partial mismatches won't crash (but print mismatches)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(" Partial state_dict load (fallback).")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print(" TFT weights loaded fully from tft.pth.")
        model.eval()
        return model, training_dataset

    # nothing found
    raise FileNotFoundError("No TFT checkpoint or tft.pth found in saved models / checkpoints. Train the TFT first.")



# === Predict LSTM ===
def predict_next_day_lstm():
    """Predict next-day price using shared scalers. Case-insensitive feature alignment."""
    try:
        import random
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model, feature_cols, seq_len = load_lstm_model()

        # load shared scalers
        with open(SAVED_MODELS_DIR / "shared_feature_scaler.pkl", "rb") as f:
            feature_scaler = pickle.load(f)
        with open(SAVED_MODELS_DIR / "shared_target_scaler.pkl", "rb") as f:
            target_scaler = pickle.load(f)

        df = safe_read_features()

        # determine close column regardless of case
        close_col = _col_exists_anycase(df, "Close")
        if close_col is None:
            raise KeyError("'Close' column missing — cannot recreate derived features.")

        # create canonical derived features with the exact names your models expect (capitalized style)
        df["Close_diff"] = df[close_col].diff().fillna(0)
        df["Close_pct_change"] = df[close_col].pct_change().fillna(0)
        df["Close_rolling_mean_5"] = df[close_col].rolling(5, min_periods=1).mean()
        df["Close_rolling_std_5"] = df[close_col].rolling(5, min_periods=1).std().fillna(0)

        # Align features to feature_cols (case-insensitive)
        df_aligned, created, missing = align_features_with_df(df, feature_cols)
        if missing:
            # try to be helpful: lowercase variants
            missing_lower = [m for m in missing if _col_exists_anycase(df, m.lower())]
            if missing_lower:
                for m in missing_lower:
                    df_aligned[m] = df[_col_exists_anycase(df, m.lower())]
                    missing.remove(m)
            # if still missing, raise
            if missing:
                raise ValueError(f"Feature columns missing from data (LSTM): {missing}")

        # Ensure numeric and no-nans
        for c in feature_cols:
            df_aligned[c] = pd.to_numeric(df_aligned[c], errors="coerce").fillna(0.0)

        # Prepare sequence input (no re-fitting scaler; assume features already scaled at training step)
        X = df_aligned[feature_cols].values
        if len(X) < seq_len:
            raise ValueError(f"Not enough data for sequence: need {seq_len}, have {len(X)}")

        # If scaler expected feature order, ensure it's the same; feature_scaler.feature_names_in_ may exist
        if hasattr(feature_scaler, "feature_names_in_"):
            ordered = [f for f in feature_scaler.feature_names_in_ if f in feature_cols]
            if len(ordered) == len(feature_cols):
                X = df_aligned[ordered].values

        # NOTE: The training saved X was already scaled; if you saved shared_feature_scaler.pkl during training,
        # and the current df_aligned is raw (not scaled), we need to scale the input:
        try:
            X_scaled = feature_scaler.transform(X)
        except Exception:
            # If already scaled (unexpected), just use X
            X_scaled = X

        sequence = np.expand_dims(X_scaled[-seq_len:], axis=0).astype(np.float32)

        device = get_device()
        model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).to(device)
            y_pred_scaled = model(sequence_tensor).cpu().numpy().reshape(-1, 1)

        # inverse target scaler
        predicted_price = float(target_scaler.inverse_transform(y_pred_scaled)[0][0])

        last_price = float(df[close_col].iloc[-1])
        change_pct = ((predicted_price - last_price) / last_price) * 100 if last_price != 0 else 0.0
        confidence = max(0.0, 100.0 - abs(change_pct))

        print(f"\n========== LSTM Prediction ==========")
        print(f"Predicted: ₹{predicted_price:.2f} | Last: ₹{last_price:.2f}")
        print(f"Change: {change_pct:.2f}% | Confidence: {confidence:.2f}%")
        print("=====================================\n")

        return {
            "predicted_price": predicted_price,
            "last_price": last_price,
            "predicted_change_pct": change_pct,
            "confidence": confidence,
            "uncertainty": abs(change_pct),
        }

    except Exception as e:
        print(f"Error predicting with LSTM: {e}")
        raise


def predict_next_day_tft():
    """
    Predict next-day closing price using the trained TFT model.
    Reconstructs derived features exactly as training expects, scales using shared scalers,
    and uses stock_data.csv for the authoritative last_price.
    """
    try:
        import random
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.use_deterministic_algorithms(True)

        model, training_dataset = load_tft_model()

        # load shared scalers & feature list
        with open(SAVED_MODELS_DIR / "shared_feature_scaler.pkl", "rb") as f:
            feature_scaler = pickle.load(f)
        with open(SAVED_MODELS_DIR / "shared_target_scaler.pkl", "rb") as f:
            target_scaler = pickle.load(f)
        with open(SAVED_MODELS_DIR / "shared_features.pkl", "rb") as f:
            feature_cols = pickle.load(f)

        # read the authoritative stock data for last_price
        stock_path = get_project_root() / "data" / "stock_data.csv"
        if stock_path.exists():
            stock_df = pd.read_csv(stock_path)
            # ensure date and sorting
            if "date" in stock_df.columns:
                stock_df["date"] = pd.to_datetime(stock_df["date"])
                stock_df = stock_df.sort_values("date").reset_index(drop=True)
            # try common close names
            if "Close" not in stock_df.columns and "close" in stock_df.columns:
                stock_df["Close"] = stock_df["close"]
            last_price = float(pd.to_numeric(stock_df["Close"].iloc[-1], errors="coerce"))
        else:
            last_price = None  # fallback if no stock file present

        # load features dataset (for model inputs)
        df = safe_read_features()  # use the helper that parses dates and ensures column strings
        # ensure date col, time_idx, group_id exist for TFT dataset
        if "date" not in df.columns:
            raise KeyError("'date' column missing from features file.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days
        df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

        # ensure canonical derived features exist
        close_col = _col_exists_anycase(df, "Close")
        if close_col is None:
            raise KeyError("'Close' column missing — cannot recreate derived features.")
        df["Close_diff"] = df[close_col].diff().fillna(0)
        df["Close_pct_change"] = df[close_col].pct_change().fillna(0)
        df["Close_rolling_mean_5"] = df[close_col].rolling(5, min_periods=1).mean()
        df["Close_rolling_std_5"] = df[close_col].rolling(5, min_periods=1).std().fillna(0)

        # align features (case-insensitive + aliasing)
        df_aligned, created, missing = align_features_with_df(df, feature_cols)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # ensure numeric and scale
        for c in feature_cols:
            df_aligned[c] = pd.to_numeric(df_aligned[c], errors="coerce").fillna(0.0)

        # attempt to scale using shared scaler (if appropriate)
        try:
            X_scaled = feature_scaler.transform(df_aligned[feature_cols].values)
            # replace in df_aligned (TimeSeriesDataSet expects columns to be present)
            df_aligned[feature_cols] = X_scaled
        except Exception:
            # if transform fails, assume features already scaled in CSV (rare)
            pass

        # add scaled target for dataset (if required)
        try:
            df_aligned["Close_scaled"] = target_scaler.transform(df_aligned[close_col].values.reshape(-1, 1))
        except Exception:
            # ignore if not necessary
            pass

        # build prediction dataset and dataloader
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, df_aligned, predict=True, stop_randomization=True
        )
        prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        # model inference
        model.eval()
        with torch.no_grad():
            preds = model.predict(prediction_dataloader, return_y=False)
        # normalize preds to numpy
        if isinstance(preds, torch.Tensor):
            preds_np = preds.detach().cpu().numpy().reshape(-1, 1)
        else:
            preds_np = np.array(preds).reshape(-1, 1)

        predicted_scaled = preds_np[-1].reshape(1, -1)
        # inverse transform predicted price into original scale using shared target scaler
        predicted_price = float(target_scaler.inverse_transform(predicted_scaled)[0][0])

        # if we couldn't find a stock_data last_price earlier, read it from features as fallback
        if last_price is None:
            last_price = float(pd.to_numeric(df_aligned[close_col].iloc[-1], errors="coerce"))

        price_change_pct = ((predicted_price - last_price) / last_price) * 100 if last_price != 0 else 0.0
        confidence = max(0.0, 100.0 - abs(price_change_pct))

        print(f"\n========== TFT Prediction ==========")
        print(f"Predicted: ₹{predicted_price:.2f} | Last: ₹{last_price:.2f}")
        print(f"Change: {price_change_pct:.2f}% | Confidence: {confidence:.2f}%")
        print("===================================\n")

        return {
            "predicted_price": predicted_price,
            "last_price": last_price,
            "predicted_change_pct": price_change_pct,
            "confidence": confidence,
            "uncertainty": abs(price_change_pct),
        }

    except Exception as e:
        print(f"Error predicting with TFT: {e}")
        import traceback
        traceback.print_exc()
        raise



# === Entry point ===
def predict_next_day(model_name="LSTM"):
    if model_name.upper() == "LSTM":
        return predict_next_day_lstm()
    elif model_name.upper() == "TFT":
        return predict_next_day_tft()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'LSTM' or 'TFT'.")


if __name__ == "__main__":
    print("Testing LSTM prediction...")
    try:
        print(predict_next_day("LSTM"))
    except Exception as e:
        print(f"LSTM prediction failed: {e}")

    print("\nTesting TFT prediction...")
    try:
        print(predict_next_day("TFT"))
    except Exception as e:
        print(f"TFT prediction failed: {e}")
