# src/eval/evaluate_models.py
"""
Evaluate LSTM, TFT, and Ensemble (optional) and save:
 - metrics JSONs
 - summary CSV
 - plots (actual vs pred, residuals, residual histogram)
 - feature importance (TFT interpret or correlation heatmap)
 - heuristic LLM-style explanations JSON
 - runtime & param counts
"""

import os
import time
import json
import math
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from scipy.stats import pearsonr

import torch
import pickle

# try imports for pytorch-forecasting; TFT won't run if missing
try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
except Exception:
    TimeSeriesDataSet = None
    TemporalFusionTransformer = None

# --- Project root detection ---
CURRENT_FILE = Path(__file__).resolve()

# Adjust the number of `.parent` depending on your folder structure:
# Here: src/evaluation/evaluate_models.py -> PROJECT_ROOT points to IPD/
PROJECT_ROOT = CURRENT_FILE.parents[2]

# --- Data paths ---
DATA_PATH = PROJECT_ROOT / "data" / "features_enhanced.csv"
STOCK_DATA_PATH = PROJECT_ROOT / "data" / "stock_data.csv"

# --- Model paths ---
SAVED_DIR = PROJECT_ROOT / "models" / "saved_models"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

# --- Output paths ---
OUT_METRICS_DIR = PROJECT_ROOT / "models" / "evaluation"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Make sure output folders exist
OUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Config (adjust if you trained with a different seq_len)
SEQ_LEN = 30
BATCH_SIZE = 256  # for inference

def load_shared_scalers():
    with open(SAVED_DIR / "shared_feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(SAVED_DIR / "shared_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    with open(SAVED_DIR / "shared_features.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return feature_scaler, target_scaler, feature_cols

# Optional: debug prints
print("Project root:", PROJECT_ROOT)
print("Saved models dir:", SAVED_DIR)


# -------------------------
# Utility metrics
# -------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, prev: np.ndarray) -> float:
    """
    percent where sign(y_pred - prev) == sign(y_true - prev)
    prev is previous day's close aligned to y_true
    """
    def sign_arr(a):
        return np.sign(a)
    pred_dir = sign_arr(y_pred - prev)
    true_dir = sign_arr(y_true - prev)
    correct = (pred_dir == true_dir).sum()
    return 100.0 * correct / len(y_true)


def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -------------------------
# Prepare LSTM validation data
# -------------------------
def build_lstm_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = SEQ_LEN, test_frac: float = 0.2):
    """
    Build sequences and targets using shared scalers already applied or apply them here.
    Returns: X_train, X_val, y_train, y_val, prev_val_close (array aligned with y_val)
    """
    # ensure correct ordering
    df = df.sort_values("date").reset_index(drop=True)

    # raw features (ensure all exist)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from data (LSTM): {missing}")

    X = df[feature_cols].astype(float).values
    y = pd.to_numeric(df["Close"], errors="coerce").values.reshape(-1, 1)

    # create sequences
    sequences = []
    targets = []
    prev_closes = []
    for i in range(seq_len, len(X)):
        sequences.append(X[i - seq_len:i])
        targets.append(y[i, 0])
        # prev close corresponds to t-1 value (for directional accuracy)
        prev_closes.append(y[i - 1, 0])

    sequences = np.array(sequences)  # (N, seq_len, n_features)
    targets = np.array(targets)
    prev_closes = np.array(prev_closes)

    split = int(len(sequences) * (1 - test_frac))
    X_train, X_val = sequences[:split], sequences[split:]
    y_train, y_val = targets[:split], targets[split:]
    prev_train, prev_val = prev_closes[:split], prev_closes[split:]

    return X_train, X_val, y_train, y_val, prev_val


# -------------------------
# LSTM predict helper
# -------------------------
def load_lstm_model_for_eval(input_size: int):
    """
    Try to load a best checkpoint from CHECKPOINT_DIR; fall back to saved state dict lstm.pt
    Returns: model (torch.nn.Module), param_count
    """
    # lazy-import LSTMModel class from your training module
    try:
        from src.training.train_lstm import LSTMModel
    except Exception as e:
        raise ImportError(f"Cannot import LSTMModel from training module: {e}")

    # find checkpoint
    ckpts = list(CHECKPOINT_DIR.glob("lstm*.ckpt"))
    device = torch.device("cpu")
    if ckpts:
        ckpt_path = ckpts[0]
        ckpt = torch.load(ckpt_path, map_location=device)
        # instantiate model using hyper-parameters if present
        hparams = ckpt.get("hyper_parameters", {})
        if "input_size" not in hparams:
            hparams["input_size"] = input_size
        else:
            # sometimes names differ
            hparams["input_size"] = input_size
        model = LSTMModel(**hparams)
        # load state
        try:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        except Exception:
            # fallback to partial load
            sd = ckpt["state_dict"]
            model_sd = model.state_dict()
            compatible = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            model_sd.update(compatible)
            model.load_state_dict(model_sd, strict=False)
    else:
        # fall back to saved model file (lstm.pt) which may be state_dict
        p = SAVED_DIR / "lstm.pt"
        if not p.exists():
            raise FileNotFoundError("No LSTM checkpoint or saved model found.")
        # need to instantiate model; try default hyperparams
        model = LSTMModel(input_size=input_size)
        sd = torch.load(p, map_location=device)
        # try multiple formats
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            model_sd = model.state_dict()
            compatible = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            model_sd.update(compatible)
            model.load_state_dict(model_sd, strict=False)

    model.to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    return model, param_count


def predict_lstm_on_val(model, X_val: np.ndarray, feature_scaler, target_scaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_val: (N, seq_len, n_features)
    Returns y_pred (unscaled), y_true (unscaled)
    """
    device = torch.device("cpu")
    # If features are already scaled in CSV, skip scaling here. We assume shared scaler expects raw inputs.
    # Try to scale inputs if scaler has feature_names_in_ attribute and shapes match
    try:
        n_features = X_val.shape[2]
        # reshape for transform: (N*seq_len, n_features)
        flat = X_val.reshape(-1, n_features)
        flat_scaled = feature_scaler.transform(flat)
        X_scaled = flat_scaled.reshape(X_val.shape)
    except Exception:
        X_scaled = X_val  # assume already scaled

    preds_scaled = []
    batch_size = BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.FloatTensor(X_scaled[i : i + batch_size])
            out = model(batch)
            out_np = out.cpu().numpy().reshape(-1, 1)
            preds_scaled.append(out_np)
    preds_scaled = np.vstack(preds_scaled)
    # inverse scale predicted y
    try:
        y_pred = target_scaler.inverse_transform(preds_scaled)
    except Exception:
        # if target scaler can't invert (maybe predictions already in target scale)
        y_pred = preds_scaled

    # build y_true from X_val's final target: we must reconstruct from original data elsewhere
    # For this function we expect caller to pass y_val separately; we'll return preds only
    return y_pred.reshape(-1)


# -------------------------
# TFT evaluation helpers
# -------------------------
def load_tft_for_eval():
    if TimeSeriesDataSet is None or TemporalFusionTransformer is None:
        raise ImportError("pytorch-forecasting not available; cannot evaluate TFT here.")
    # load training dataset pickle
    p = SAVED_DIR / "tft_training_dataset.pkl"
    model_p = SAVED_DIR / "tft.pth"
    if not p.exists() or not model_p.exists():
        raise FileNotFoundError("Missing TFT dataset pickle or model file.")
    with open(p, "rb") as f:
        training = pickle.load(f)
    # reconstruct model from dataset
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=training.hparams.get("hidden_size", 64) if hasattr(training, "hparams") else 64,
        attention_head_size=training.hparams.get("attention_head_size", 2) if hasattr(training, "hparams") else 2,
        dropout=0.1,
        hidden_continuous_size=64,
        output_size=1,
        loss=None,
    )
    # load weights
    weights = torch.load(model_p, map_location="cpu")
    if isinstance(weights, dict) and "model_state_dict" in weights:
        sd = weights["model_state_dict"]
    else:
        sd = weights
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print("Warning: partial or mismatched TFT state dict load:", e)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model, training


def predict_tft_on_val(model, training_dataset, df: pd.DataFrame):
    """
    Build validation dataset from df using training_dataset config, then predict.
    Returns y_pred (unscaled), y_true (unscaled), prev_close aligned for DA.
    """
    # create validation dataset by reusing the same training dataset config:
    # we will create a validation df by using last portion equal to training_dataset.max_prediction_length + ... simplest is to build from entire df but set predict=True
    val_ds = TimeSeriesDataSet.from_dataset(training_dataset, df, predict=False, stop_randomization=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    # model.predict returns numpy or tensor; use pytorch-forecasting predict
    preds = model.predict(val_dl)
    # preds shape likely (N, 1) or (N,) depending; force to numpy flat
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy().reshape(-1)
    else:
        preds_np = np.array(preds).reshape(-1)
    # y_true: fetch target values from val_ds: val_ds.to_dataset? easier: get y from val_ds
    y_true = []
    prev_close = []
    # Iterate through validation dataloader to gather ground truth and prev close
    for batch in val_dl:
        # pytorch-forecasting returns batch as dictionary or tuple; training_dataset expects (x, y) for our wrapper; but here dataloader yields tensors.
        # We pull target from batch[1] if tuple
        try:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                y = batch[1]
                # y shape (batch_size, prediction_length, 1) - we use last dim
                y = y[:, 0, 0].cpu().numpy()
                y_true.extend(y.tolist())
                # prev close: attempt to read encoder last close in batch[0] using variable name
                # best-effort: training_dataset has variable_names mapping; fallback to NaN
                prev_close.extend([np.nan] * len(y))
            else:
                # unknown batch format
                prev_close.extend([np.nan] * len(preds_np))
        except Exception:
            prev_close.extend([np.nan] * len(preds_np))
    y_true = np.array(y_true)
    prev_close = np.array(prev_close)
    # If sizes mismatch, trim or pad
    n = min(len(preds_np), len(y_true))
    return preds_np[:n], y_true[:n], prev_close[:n], val_ds


# -------------------------
# Plot helpers
# -------------------------
def plot_actual_vs_pred(dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, outpath: Path, title: str, zoom_n: int = 200):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", linewidth=1)
    plt.plot(dates, y_pred, label="Predicted", linewidth=1)
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    # zoom
    plt.figure(figsize=(12, 5))
    plt.plot(dates[-zoom_n:], y_true[-zoom_n:], label="Actual", linewidth=1)
    plt.plot(dates[-zoom_n:], y_pred[-zoom_n:], label="Predicted", linewidth=1)
    plt.legend()
    plt.title(f"{title} (last {zoom_n} samples)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(outpath.with_name(outpath.stem + "_zoom.png"))
    plt.close()


def plot_residuals(dates: np.ndarray, residuals: np.ndarray, outpath: Path, title: str):
    plt.figure(figsize=(12, 4))
    plt.plot(dates, residuals, linewidth=1)
    plt.title(title + " - Residuals")
    plt.xlabel("Date")
    plt.ylabel("Residual (Pred - True)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50)
    plt.title(title + " - Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath.with_name(outpath.stem + "_hist.png"))
    plt.close()


# -------------------------
# Feature importance / correlation
# -------------------------
def feature_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str], outpath: Path):
    # compute Pearson correlation of each feature with Close
    corrs = {}
    for c in feature_cols:
        try:
            a = pd.to_numeric(df[c], errors="coerce").fillna(0).values
            b = pd.to_numeric(df["Close"], errors="coerce").fillna(0).values
            if len(a) != len(b):
                mn = min(len(a), len(b))
                a = a[-mn:]
                b = b[-mn:]
            corrs[c] = pearsonr(a, b)[0]
        except Exception:
            corrs[c] = 0.0
    # sort top 20
    items = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:40]
    names, vals = zip(*items)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.title("Top feature correlations with Close (Pearson r)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# -------------------------
# LLM-style heuristic explanations
# -------------------------
def generate_llm_explanations(dates: List[str], y_true: np.ndarray, y_pred: np.ndarray, n_examples: int = 8):
    """
    Create simple heuristic explanations saved as JSON. Replace with real LLM calls if desired.
    """
    out = []
    N = len(y_true)
    idxs = np.linspace(0, N - 1, min(n_examples, N)).astype(int)
    for i in idxs:
        true = y_true[i]
        pred = y_pred[i]
        change_true = "up" if true > (0 if i == 0 else y_true[i - 1]) else "down"
        change_pred = "up" if pred > (0 if i == 0 else y_pred[i - 1]) else "down"
        explanation = (
            f"Model predicted {change_pred} because recent short-term momentum and moving averages "
            f"indicate a shift. Absolute error = {pred - true:.4f}."
        )
        out.append(
            {
                "date": str(dates[i])[:10],
                "true_change": change_true,
                "predicted_change": change_pred,
                "true_price": float(true),
                "pred_price": float(pred),
                "explanation": explanation,
            }
        )
    return out


# -------------------------
# Main evaluation flow
# -------------------------
def main():
    t0 = time.perf_counter()

    # load shared scalers + features
    feature_scaler, target_scaler, feature_cols = load_shared_scalers()
    print("Loaded shared scalers and feature list:", len(feature_cols), "features")

    # load features dataframe
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Ensure derived columns exist for backward compatibility ---
    if "Close" not in df.columns and "close" in df.columns:
        df["Close"] = df["close"]

    if "close_lag_1" not in df.columns:
        df["close_lag_1"] = df["Close"].shift(1)
    if "close_lag_5" not in df.columns:
        df["close_lag_5"] = df["Close"].shift(5)
    if "Close_diff" not in df.columns:
        df["Close_diff"] = df["Close"].diff()
    if "Close_pct_change" not in df.columns:
        df["Close_pct_change"] = df["Close"].pct_change()
    if "Close_rolling_mean_5" not in df.columns:
        df["Close_rolling_mean_5"] = df["Close"].rolling(5, min_periods=1).mean()
    if "Close_rolling_std_5" not in df.columns:
        df["Close_rolling_std_5"] = df["Close"].rolling(5, min_periods=1).std()

    df = df.fillna(0)


    # --- Build LSTM sequences and validation set ---
    print("Preparing LSTM sequences...")
    X_train, X_val, y_train, y_val, prev_val = build_lstm_sequences(df, feature_cols, seq_len=SEQ_LEN, test_frac=0.2)
    val_dates = df["date"].iloc[SEQ_LEN + len(X_train) : SEQ_LEN + len(X_train) + len(X_val)].values

    # --- Load LSTM model and predict ---
    print("Loading LSTM model...")
    lstm_model, lstm_params = load_lstm_model_for_eval(input_size=len(feature_cols))
    t_lstm_start = time.perf_counter()
    y_pred_lstm = predict_lstm_on_val(lstm_model, X_val, feature_scaler, target_scaler)
    t_lstm = time.perf_counter() - t_lstm_start

    # align y_true
    y_true_val = y_val  # raw prices
    prev_close_val = prev_val

    # compute metrics
    mse = mean_squared_error(y_true_val, y_pred_lstm)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_val, y_pred_lstm)
    mape = mean_absolute_percentage_error(y_true_val, y_pred_lstm) * 100
    r2 = r2_score(y_true_val, y_pred_lstm)
    da = directional_accuracy(y_true_val, y_pred_lstm, prev_close_val)

    lstm_metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE_percent": float(mape),
        "R2": float(r2),
        "DA_percent": float(da),
        "param_count": int(lstm_params),
        "inference_time_s": float(t_lstm),
    }
    save_json(OUT_METRICS_DIR / "lstm_metrics.json", lstm_metrics)
    print("Saved LSTM metrics:", lstm_metrics)

    # plots for LSTM
    print("Plotting LSTM charts...")
    plot_actual_vs_pred(val_dates, y_true_val, y_pred_lstm, PLOTS_DIR / "lstm_actual_vs_pred.png", "LSTM Actual vs Predicted")
    plot_residuals(val_dates, (y_pred_lstm - y_true_val), PLOTS_DIR / "lstm_residuals.png", "LSTM")

    # -------------------------
    # TFT evaluation
    # -------------------------
    tft_metrics = {}
    tft_preds = None
    try:
        if TimeSeriesDataSet is None:
            raise ImportError("pytorch-forecasting not installed.")
        print("Loading TFT model and dataset...")
        tft_model, training_dataset = load_tft_for_eval()
        # create df_for_tft: must include the same columns training expected (feature_cols + time_idx + group_id + Close)
        # align column names and types; ensure 'time_idx' and 'group_id' in df
        df_tft = df.copy()
        df_tft["time_idx"] = (df_tft["date"] - df_tft["date"].min()).dt.days
        df_tft["group_id"] = df_tft.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")
        # scale numeric inputs using shared scaler (if possible)
        try:
            df_tft[feature_cols] = feature_scaler.transform(df_tft[feature_cols].astype(float))
        except Exception:
            pass
        # add scaled target
        try:
            df_tft["Close_scaled"] = target_scaler.transform(df_tft["Close"].values.reshape(-1, 1))
        except Exception:
            pass

        t0_tft = time.perf_counter()
        y_pred_tft, y_true_tft, prev_tft, val_ds = predict_tft_on_val(tft_model, training_dataset, df_tft)
        t_tft = time.perf_counter() - t0_tft

        if len(y_true_tft) == 0:
            raise RuntimeError("TFT validation yielded zero-length y_true. Check training_dataset and features.")

        # For prev closes, try to derive from df aligned to val indices - fallback to NaN array
        prev_close_tft = np.where(np.isnan(prev_tft), np.roll(y_true_tft, 1), prev_tft)
        mse_t = mean_squared_error(y_true_tft, y_pred_tft)
        rmse_t = math.sqrt(mse_t)
        mae_t = mean_absolute_error(y_true_tft, y_pred_tft)
        mape_t = mean_absolute_percentage_error(y_true_tft, y_pred_tft) * 100
        r2_t = r2_score(y_true_tft, y_pred_tft)
        da_t = directional_accuracy(y_true_tft, y_pred_tft, prev_close_tft)

        tft_metrics = {
            "MSE": float(mse_t),
            "RMSE": float(rmse_t),
            "MAE": float(mae_t),
            "MAPE_percent": float(mape_t),
            "R2": float(r2_t),
            "DA_percent": float(da_t),
            "inference_time_s": float(t_tft),
        }
        save_json(OUT_METRICS_DIR / "tft_metrics.json", tft_metrics)
        print("Saved TFT metrics:", tft_metrics)

        # plots for TFT
        # use val_ds to get dates if possible; otherwise fallback to tail of df
        try:
            # attempt to extract index mapping from val_ds or df_tft
            val_start_idx = len(df_tft) - len(y_true_tft)
            val_dates_tft = df_tft["date"].iloc[val_start_idx: val_start_idx + len(y_true_tft)].values
        except Exception:
            val_dates_tft = df_tft["date"].tail(len(y_true_tft)).values

        plot_actual_vs_pred(val_dates_tft, y_true_tft, y_pred_tft, PLOTS_DIR / "tft_actual_vs_pred.png", "TFT Actual vs Predicted")
        plot_residuals(val_dates_tft, (y_pred_tft - y_true_tft), PLOTS_DIR / "tft_residuals.png", "TFT")

        # feature importance (try interpret_output, else correlation heatmap)
        try:
            print("Attempting TFT interpret_output for variable importance (may be slow)...")
            # pytorch-forecasting interpret API: model.interpret_output(val_dataloader) or model.interpret_output(val_ds)
            try:
                imp = tft_model.interpret_output(val_ds)
            except Exception:
                # fallback: use training_dataset and its dataloader
                val_dl = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
                imp = tft_model.interpret_output(val_dl)
            # interpret_output returns importance dict or DataFrame; save top features if present
            # try to save to JSON or create barplot
            if isinstance(imp, dict):
                # try to get variable_importance key
                var_imp = imp.get("variable_importance", imp)
            else:
                var_imp = imp
            # try to convert to series
            try:
                if hasattr(var_imp, "sort_values"):
                    var_imp = var_imp.sort_values(ascending=False)
                    top = var_imp.head(20)
                    plt.figure(figsize=(8, 6))
                    top.plot.barh()
                    plt.gca().invert_yaxis()
                    plt.title("TFT Variable Importance")
                    plt.tight_layout()
                    plt.savefig(PLOTS_DIR / "feature_importance.png")
                    plt.close()
                else:
                    raise Exception("Unexpected interpret_output format")
            except Exception:
                # fallback to correlation heatmap
                feature_correlation_heatmap(df, feature_cols, PLOTS_DIR / "feature_correlation.png")
        except Exception as e:
            print("TFT interpret_output failed or unavailable:", e)
            feature_correlation_heatmap(df, feature_cols, PLOTS_DIR / "feature_correlation.png")

    except Exception as e:
        print("TFT evaluation skipped or failed:", e)
        tft_metrics = {}
        y_pred_tft, y_true_tft = np.array([]), np.array([])

    # -------------------------
    # Ensemble (simple average) - optional
    # -------------------------
    ensemble_metrics = {}
    try:
        if len(y_pred_lstm) and len(y_pred_tft) and len(y_true_tft):
            # align lengths by using shortest
            n = min(len(y_pred_lstm), len(y_pred_tft), len(y_true_tft))
            y_pred_ens = (y_pred_lstm[-n:] + y_pred_tft[-n:]) / 2.0
            y_true_ens = y_true_tft[-n:]
            prev_ens = np.roll(y_true_ens, 1)
            mse_e = mean_squared_error(y_true_ens, y_pred_ens)
            rmse_e = math.sqrt(mse_e)
            mae_e = mean_absolute_error(y_true_ens, y_pred_ens)
            mape_e = mean_absolute_percentage_error(y_true_ens, y_pred_ens) * 100
            r2_e = r2_score(y_true_ens, y_pred_ens)
            da_e = directional_accuracy(y_true_ens, y_pred_ens, prev_ens)
            ensemble_metrics = {
                "MSE": float(mse_e),
                "RMSE": float(rmse_e),
                "MAE": float(mae_e),
                "MAPE_percent": float(mape_e),
                "R2": float(r2_e),
                "DA_percent": float(da_e),
            }
            save_json(OUT_METRICS_DIR / "ensemble_metrics.json", ensemble_metrics)
            plot_actual_vs_pred(val_dates_tft[-n:], y_true_ens, y_pred_ens, PLOTS_DIR / "ensemble_actual_vs_pred.png", "Ensemble Actual vs Predicted")
            plot_residuals(val_dates_tft[-n:], (y_pred_ens - y_true_ens), PLOTS_DIR / "ensemble_residuals.png", "Ensemble")
            print("Saved Ensemble metrics")
    except Exception as e:
        print("Ensemble evaluation skipped:", e)

    # -------------------------
    # Save summary CSV
    # -------------------------
    rows = []
    if lstm_metrics:
        rows.append({"Model": "LSTM", "RMSE": lstm_metrics["RMSE"], "MAE": lstm_metrics["MAE"], "MAPE_percent": lstm_metrics["MAPE_percent"], "R2": lstm_metrics["R2"], "DA_percent": lstm_metrics["DA_percent"]})
    if tft_metrics:
        rows.append({"Model": "TFT", "RMSE": tft_metrics.get("RMSE"), "MAE": tft_metrics.get("MAE"), "MAPE_percent": tft_metrics.get("MAPE_percent"), "R2": tft_metrics.get("R2"), "DA_percent": tft_metrics.get("DA_percent")})
    if ensemble_metrics:
        rows.append({"Model": "Ensemble", "RMSE": ensemble_metrics.get("RMSE"), "MAE": ensemble_metrics.get("MAE"), "MAPE_percent": ensemble_metrics.get("MAPE_percent"), "R2": ensemble_metrics.get("R2"), "DA_percent": ensemble_metrics.get("DA_percent")})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_METRICS_DIR / "summary_metrics.csv", index=False)
    print("Saved summary CSV:", OUT_METRICS_DIR / "summary_metrics.csv")

    # Save individual metrics JSONs (already saved for LSTM/TFT); ensure ensemble saved
    if ensemble_metrics:
        save_json(OUT_METRICS_DIR / "ensemble_metrics.json", ensemble_metrics)

    # LLM-style explanations (heuristic)
    print("Generating heuristic LLM-style explanations...")
    # prefer use TFT or LSTM arrays where available
    if len(y_true_val):
        dates_for_examples = val_dates
        ex = generate_llm_explanations(dates_for_examples, y_true_val, y_pred_lstm, n_examples=10)

        from src.utils.helpers import ensure_dir
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]
        RESULTS_DIR = project_root / "results"
        ensure_dir(RESULTS_DIR)

        save_json(RESULTS_DIR / "llm_explanations.json", ex)
        print(f"Saved heuristic LLM explanations → {RESULTS_DIR / 'llm_explanations.json'}")


    # runtime and dataset snapshot
    total_time = time.perf_counter() - t0
    runtime_info = {
        "total_eval_seconds": total_time,
        "lstm_inference_seconds": lstm_metrics.get("inference_time_s"),
        "tft_inference_seconds": tft_metrics.get("inference_time_s"),
        "dataset_rows": len(df),
        "feature_count": len(feature_cols),
        "seq_len": SEQ_LEN,
        "date_range": [str(df["date"].min())[:10], str(df["date"].max())[:10]],
    }
    save_json(OUT_METRICS_DIR / "runtime_info.json", runtime_info)
    print("Saved runtime/info")

    print("Evaluation finished. Outputs written to:")
    print(" -", OUT_METRICS_DIR)
    print(" -", PLOTS_DIR)


if __name__ == "__main__":
    main()
