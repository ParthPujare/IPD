"""
Evaluate LSTM model on validation set and compute metrics for the paper.

Usage:
    python -m src.evaluation.evaluate_lstm
"""

import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from math import sqrt
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, get_device, ensure_dir

# Try importing LSTM model class
try:
    from src.training.train_lstm import LSTMModel
except Exception as e:
    raise ImportError("Could not import LSTMModel from src.training.train_lstm: " + str(e))

# === Paths ===
ROOT = get_project_root()
FEATURES_DATA_PATH = ROOT / "data" / "features_enhanced.csv"
SAVED_MODELS_DIR = ROOT / "models" / "saved_models"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
OUT_DIR = ROOT / "models" / "evaluation"
ensure_dir(OUT_DIR)

# === Metadata files ===
FEATURE_SCALER_FILE = SAVED_MODELS_DIR / "lstm_feature_scaler.pkl"
TARGET_SCALER_FILE = SAVED_MODELS_DIR / "lstm_target_scaler.pkl"
FEATURES_FILE = SAVED_MODELS_DIR / "lstm_features.pkl"

# === Parameters ===
SEQ_LEN = 30
TEST_SIZE = 0.2
BATCH_SIZE = 64


def load_latest_checkpoint():
    ckpts = list(CHECKPOINT_DIR.glob("lstm*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No LSTM checkpoints found in {CHECKPOINT_DIR}")
    latest = sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
    return str(latest)


def build_sequences(df, feature_cols, feature_scaler, target_scaler, seq_len=SEQ_LEN):
    """Rebuild sequences with proper scaling for both features and target."""
    X = df[feature_cols].values
    y = df["Close"].values.reshape(-1, 1)

    # Scale features and target exactly as done in training
    X_scaled = feature_scaler.transform(X)
    y_scaled = target_scaler.transform(y)

    sequences, targets = [], []
    for i in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[i - seq_len:i])
        targets.append(y_scaled[i])  # scaled target
    return np.array(sequences), np.array(targets).squeeze()


def main():
    print("Loading data and artifacts...")

    # Load data
    df = pd.read_csv(FEATURES_DATA_PATH).sort_values("date").reset_index(drop=True)

    # Load metadata
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} not found.")
    with open(FEATURES_FILE, "rb") as f:
        feature_cols = pickle.load(f)

    if not FEATURE_SCALER_FILE.exists() or not TARGET_SCALER_FILE.exists():
        raise FileNotFoundError("Missing scaler files in saved_models/")

    with open(FEATURE_SCALER_FILE, "rb") as f:
        feature_scaler = pickle.load(f)
    with open(TARGET_SCALER_FILE, "rb") as f:
        target_scaler = pickle.load(f)

    # Build sequences (properly scaled)
    X_all, y_all = build_sequences(df, feature_cols, feature_scaler, target_scaler, seq_len=SEQ_LEN)
    if len(X_all) == 0:
        raise ValueError("No sequences available. Check SEQ_LEN and dataset length.")

    # Split train/validation same as training
    split_idx = int(len(X_all) * (1 - TEST_SIZE))
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]

    print(f"Total sequences: {len(X_all)} | Validation: {len(X_val)}")

    # Load latest model checkpoint
    ckpt_path = load_latest_checkpoint()
    print("Loading model checkpoint:", ckpt_path)
    device = get_device()
    model = LSTMModel.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # Predict in batches
    preds_scaled = []
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            batch = X_val[i:i+BATCH_SIZE]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            yhat = model(batch_tensor)
            preds_scaled.append(yhat.detach().cpu().numpy().reshape(-1, 1))

    preds_scaled = np.vstack(preds_scaled).squeeze()

    # Inverse scale predictions and targets
    y_val_orig = target_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
    preds_orig = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

    # Compute metrics
    mse = mean_squared_error(y_val_orig, preds_orig)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_val_orig, preds_orig)
    mape = mean_absolute_percentage_error(y_val_orig, preds_orig) * 100
    r2 = r2_score(y_val_orig, preds_orig)

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape_percent": float(mape),
        "r2": float(r2),
        "n_val": int(len(y_val_orig)),
    }

    # Print summary
    print("\n=== 📈 LSTM Evaluation Report ===")
    print(f"Validation samples: {metrics['n_val']}")
    print(f"MSE:   {metrics['mse']:.4f}")
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"MAPE:  {metrics['mape_percent']:.4f}%")
    print(f"R²:    {metrics['r2']:.4f}")

    # Save outputs
    with open(OUT_DIR / "lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    df_out = pd.DataFrame({"y_true": y_val_orig, "y_pred": preds_orig})
    df_out.to_csv(OUT_DIR / "lstm_val_predictions.csv", index=False)

    print(f"\n✅ Saved metrics to {OUT_DIR / 'lstm_metrics.json'}")
    print(f"✅ Saved predictions to {OUT_DIR / 'lstm_val_predictions.csv'}")


if __name__ == "__main__":
    main()
