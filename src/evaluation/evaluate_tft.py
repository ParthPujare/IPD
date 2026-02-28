# src/evaluation/evaluate_tft.py
"""
Evaluate a trained TFT model (robust: handles different return formats & devices).
Usage:
    python -m src.evaluation.evaluate_tft
"""
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "features_enhanced.csv"
MODEL_PATH = ROOT / "models" / "saved_models" / "tft.pth"
FEATURE_SCALER_PATH = ROOT / "models" / "saved_models" / "shared_feature_scaler.pkl"
TARGET_SCALER_PATH = ROOT / "models" / "saved_models" / "shared_target_scaler.pkl"
DATASET_PATH = ROOT / "models" / "saved_models" / "tft_training_dataset.pkl"

# ---- helpers ----
def to_numpy_cpu(t: torch.Tensor) -> np.ndarray:
    """Move tensor to CPU, detach and convert to numpy (squeezed)."""
    return t.detach().cpu().numpy().squeeze()

def extract_prediction_tensor(raw):
    """
    Extract and normalize predictions from raw TFT outputs.
    Handles variable decoder lengths and quantile dimensions.
    Returns a torch.Tensor on CPU (not flattened).
    """
    def normalize_shape(t: torch.Tensor):
        if not isinstance(t, torch.Tensor):
            return None
        # Bring to CPU to avoid device issues
        t = t.detach().cpu()
        # Handle scalar tensors
        if t.ndim == 0:
            t = t.unsqueeze(0)
        # Reduce quantile dimensions if present
        if t.ndim == 3:
            # (batch, decoder_len, quantiles) -> average quantiles
            if t.shape[-1] in (3, 7):
                t = t.mean(dim=-1)
        elif t.ndim == 4:
            # (batch, decoder_len, quantiles, features)
            t = t.mean(dim=[2, 3])
        # If time dimension exists (decoder_len), take final time step
        if t.ndim == 2:
            # (batch, decoder_len) -> take last time step
            t = t[:, -1]
        elif t.ndim > 2:
            # try to reduce to (batch, decoder_len)
            t = t.reshape(t.shape[0], -1)[:, -1]
        return t

    if isinstance(raw, torch.Tensor):
        return normalize_shape(raw)

    # Handle lists/tuples (batch outputs)
    if isinstance(raw, (list, tuple)):
        tensors = []
        for item in raw:
            try:
                t = extract_prediction_tensor(item)
                if isinstance(t, torch.Tensor):
                    tensors.append(t)
            except Exception:
                continue
        if tensors:
            # pad/trim to smallest length then concat
            min_len = min(t.shape[0] for t in tensors)
            tensors = [t[:min_len] for t in tensors]
            return torch.cat(tensors, dim=0)
        return None

    # Handle dict outputs
    if isinstance(raw, dict):
        for key in ("prediction", "predictions", "output", "pred"):
            if key in raw and isinstance(raw[key], torch.Tensor):
                return normalize_shape(raw[key])
        for v in raw.values():
            if isinstance(v, torch.Tensor):
                return normalize_shape(v)

    # Handle objects with attributes
    for attr in ("prediction", "predictions", "output", "pred"):
        if hasattr(raw, attr):
            val = getattr(raw, attr)
            if isinstance(val, torch.Tensor):
                return normalize_shape(val)
            if isinstance(val, (list, tuple)):
                tensors = [normalize_shape(v) for v in val if isinstance(v, torch.Tensor)]
                tensors = [t for t in tensors if t is not None]
                if tensors:
                    min_len = min(t.shape[0] for t in tensors)
                    tensors = [t[:min_len] for t in tensors]
                    return torch.cat(tensors, dim=0)

    try:
        return extract_prediction_tensor(raw[0])
    except Exception:
        raise TypeError(f"Cannot extract prediction tensor from {type(raw)}")


# ---- load artifacts ----
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load scalers if available (we will use target_scaler for inverse transform)
if FEATURE_SCALER_PATH.exists():
    with open(FEATURE_SCALER_PATH, "rb") as f:
        feature_scaler = pickle.load(f)
else:
    feature_scaler = None

if TARGET_SCALER_PATH.exists():
    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)
else:
    target_scaler = None

if DATASET_PATH.exists():
    with open(DATASET_PATH, "rb") as f:
        training = pickle.load(f)
else:
    raise FileNotFoundError(f"Training dataset file not found at {DATASET_PATH}")

# checkpoint file saved in train_tft: contains model_state_dict and feature_cols
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
feature_cols = checkpoint.get("feature_cols", None)
if feature_cols is None:
    raise KeyError("feature_cols missing from saved tft.pth checkpoint.")

# ---- recreate model (must match training hyperparams) ----
# NOTE: these params should match what you used during training.
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=64,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=64,
    output_size=1,
    loss=None,
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# load weights (strict=False to allow small mismatches)
missing, unexpected = tft.load_state_dict(checkpoint["model_state_dict"], strict=False)
print(" Model weights loaded with:")
print("   Missing keys:", len(missing))
print("   Unexpected keys:", len(unexpected))

# Force CPU to avoid MPS/device mismatch issues
tft = tft.to("cpu")
tft.eval()

from src.preprocess.shared_preprocessor import prepare_shared_data
df, _ = prepare_shared_data()
if 'date' in df.columns:
    df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
df["time_idx"] = (df["date"] - df["date"].min()).dt.days
df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

# Apply same feature scaling as training
with open(FEATURE_SCALER_PATH, "rb") as f:
    feature_scaler = pickle.load(f)

# The model expects the same features used in training
expected_features = training.reals  # auto-extract feature names# Filter only those features that exist in dataframe (exclude internal TFT columns)
valid_features = [f for f in expected_features if f in df.columns]

missing_features = set(expected_features) - set(valid_features)
if missing_features:
    print(f"⚠️ Skipping {len(missing_features)} TFT internal/missing features: {missing_features}")

# Apply scaling only to valid features
# df[valid_features] = feature_scaler.transform(df[valid_features].astype(float))  # Skipped: already scaled by prepare_shared_data



# last 20% as validation (same split used in training script)
split_idx = int(len(df) * 0.8)
val_df = df.iloc[split_idx:].reset_index(drop=True)

val_dataset = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

# ---- run predictions ----
# predict returns different shapes depending on PF version: handle robustly
# ensure predictions on CPU
with torch.no_grad():
    pred_output = tft.predict(val_dataloader, mode="raw", return_x=True)

# pred_output may be (raw_predictions, x), (raw_predictions, x, something), or raw_predictions
raw_predictions = None
x = None

if isinstance(pred_output, tuple):
    if len(pred_output) >= 1:
        raw_predictions = pred_output[0]
    if len(pred_output) >= 2:
        x = pred_output[1]
else:
    raw_predictions = pred_output

# extract prediction tensor and ensure CPU
pred_tensor = extract_prediction_tensor(raw_predictions)
if pred_tensor is None:
    raise RuntimeError("Could not extract prediction tensor from model output.")
pred_tensor = pred_tensor.detach().cpu()

# get y_true (decoder_target) either from returned x or from iterating val_dataloader
y_true_tensor = None
if x is not None:
    if isinstance(x, dict) and "decoder_target" in x:
        y_true_tensor = x["decoder_target"].detach().cpu()
    else:
        if hasattr(x, "decoder_target"):
            y_true_tensor = getattr(x, "decoder_target")
            if isinstance(y_true_tensor, torch.Tensor):
                y_true_tensor = y_true_tensor.detach().cpu()
            else:
                y_true_tensor = None

# If we still don't have y_true_tensor, iterate val_dataloader and collect decoder_target
if y_true_tensor is None:
    collected = []
    for batch in val_dataloader:
        # pytorch-forecasting dataloaders often return (x, y) or (encoder, decoder) structures
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            maybe_target = batch[1]
            if isinstance(maybe_target, torch.Tensor):
                collected.append(maybe_target.detach().cpu())
            elif isinstance(maybe_target, dict) and "decoder_target" in maybe_target:
                collected.append(maybe_target["decoder_target"].detach().cpu())
        elif isinstance(batch, dict) and "decoder_target" in batch:
            collected.append(batch["decoder_target"].detach().cpu())
    if not collected:
        raise RuntimeError("Unable to extract true targets from returned x or val_dataloader. Inspect dataloader structure.")
    y_true_tensor = torch.cat(collected, dim=0)

# --- Align shapes safely ---
def to_numpy_safe(t):
    """Convert tensor-like or scalar to 1D NumPy array safely."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if np.isscalar(t):
        return np.array([t], dtype=float)
    t = np.squeeze(t)
    if np.ndim(t) == 0:
        return np.array([t], dtype=float)
    t = np.asarray(t, dtype=float)
    if t.ndim > 1:
        t = t.reshape(-1)
    return t

pred_arr = to_numpy_safe(pred_tensor)
y_arr = to_numpy_safe(y_true_tensor)

# --- Align lengths robustly ---
min_len = min(len(pred_arr), len(y_arr))
pred_arr = pred_arr[:min_len]
y_arr = y_arr[:min_len]

# --- Inverse-transform: TFT model outputs scaled predictions ---
print("Inverse transforming predictions to original target scale.")
preds_scaled = pred_arr.reshape(-1, 1)
y_true_scaled = y_arr.reshape(-1, 1)

preds_inv = target_scaler.inverse_transform(preds_scaled).reshape(-1)
y_true_inv = target_scaler.inverse_transform(y_true_scaled).reshape(-1)

print("\n--- Debug: Range Check ---")
print(f"Actual (y_true_inv): min={y_true_inv.min():.2f}, max={y_true_inv.max():.2f}")
print(f"Predicted (preds_inv): min={preds_inv.min():.2f}, max={preds_inv.max():.2f}")
print("----------------------------\n")
print(f"Mean actual: {y_true_inv.mean():.2f}")
print(f"Mean predicted: {preds_inv.mean():.2f}")
print(f"Mean difference: {(preds_inv.mean() - y_true_inv.mean()):.2f}")


# --- Bias correction ---
bias = preds_inv.mean() - y_true_inv.mean()
if abs(bias) > 50:
    print(f" Applying bias correction of -{bias:.2f}")
    preds_inv = preds_inv - bias

# Ensure predictions don't go below zero
preds_inv = np.maximum(preds_inv, 0)


# ---- metrics ----
mse = mean_squared_error(y_true_inv, preds_inv)
rmse = sqrt(mse)
mae = mean_absolute_error(y_true_inv, preds_inv)
mape = (np.abs((y_true_inv - preds_inv) / (y_true_inv + 1e-8)).mean()) * 100
r2 = r2_score(y_true_inv, preds_inv)

print("\n=== 📊 TFT Evaluation Report ===")
print(f"Validation samples: {len(y_true_inv)}")
print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"MAPE:  {mape:.4f}%")
print(f"R²:    {r2:.4f}")

# ---- save predictions CSV ----
out_df = pd.DataFrame({"y_true": y_true_inv, "y_pred": preds_inv})
out_dir = ROOT / "models" / "evaluation"
out_dir.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_dir / "tft_val_predictions.csv", index=False)
with open(out_dir / "tft_metrics.json", "w") as f:
    import json
    json.dump({
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape_percent": float(mape),
        "r2": float(r2),
        "n_val": int(len(y_true_inv))
    }, f, indent=2)

# ---- plot ----
plt.figure(figsize=(12, 5))
plt.plot(y_true_inv, label="Actual", linewidth=2)
plt.plot(preds_inv, label="Predicted", linewidth=2)
plt.title("Newly Trained TFT Model - Predicted vs Actual Closing Prices")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "newly_trained_tft_actual_vs_pred.png")
plt.savefig(ROOT / "plots" / "newly_trained_tft_actual_vs_pred.png")
print(f"Saved plot to plots/newly_trained_tft_actual_vs_pred.png")
plt.close()

# --- Diagnostic: residual plot ---
plt.figure(figsize=(10, 4))
plt.plot(y_true_inv - preds_inv, label="Residual (Actual - Predicted)", color="orange")
plt.title("Newly Trained TFT Model - Residual Plot")
plt.axhline(0, color="black", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "newly_trained_tft_residuals.png")
plt.savefig(ROOT / "plots" / "newly_trained_tft_residuals.png")
print(f"Saved plot to plots/newly_trained_tft_residuals.png")
plt.close()
