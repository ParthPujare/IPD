"""
Temporal Fusion Transformer (TFT) training script using pytorch-forecasting.
Trains a TFT model to predict next-day closing price with consistent preprocessing (same as LSTM).
"""

import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import pytorch_lightning as pl
from src.preprocess.shared_preprocessor import prepare_shared_data
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import sys
import os

# Environment setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if not hasattr(np, "float"):
    np.float = float  # for older code compatibility

# === Path setup ===
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir

ROOT = get_project_root()
FEATURES_DATA_PATH = ROOT / "data" / "features_enhanced.csv"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
SAVED_MODELS_DIR = ROOT / "models" / "saved_models"
ensure_dir(CHECKPOINT_DIR)
ensure_dir(SAVED_MODELS_DIR)

# TensorBoard logger
tb_logger = TensorBoardLogger("lightning_logs", name="tft_runs")


# === Consistent preprocessing with LSTM ===
# def prepare_tft_data(seq_len=30, test_size=0.2, debug=False, min_val_samples=120):
#     """
#     Prepare data for Temporal Fusion Transformer:
#     - Prefer precomputed features (Close_sma_5, Close_std_5) if present,
#       otherwise compute rolling mean/std and create canonical column names
#       Close_rolling_mean_5 and Close_rolling_std_5 so training and inference
#       use identical feature names.
#     - Maintain at least 'min_val_samples' validation samples
#     - Keep seq_len overlap before validation split for encoder context
#     """
#     df = pd.read_csv(FEATURES_DATA_PATH).sort_values("date").reset_index(drop=True)
#     df["date"] = pd.to_datetime(df["date"])
#     df["time_idx"] = (df["date"] - df["date"].min()).dt.days
#     df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

#     if debug:
#         df = df.tail(500).reset_index(drop=True)

#     target_col = "Close"
#     exclude_cols = ["date", "ticker", "sentiment_label_mode", "time_idx", "group_id", target_col]

#     # --- Ensure canonical engineered features exist (prefer CSV precomputed ones) ---
#     # Rolling mean 5
#     if "Close_rolling_mean_5" not in df.columns:
#         if "Close_sma_5" in df.columns:
#             df["Close_rolling_mean_5"] = df["Close_sma_5"]
#         else:
#             df["Close_rolling_mean_5"] = df["Close"].rolling(window=5, min_periods=1).mean()

#     # Rolling std 5
#     if "Close_rolling_std_5" not in df.columns:
#         if "Close_std_5" in df.columns:
#             df["Close_rolling_std_5"] = df["Close_std_5"]
#         else:
#             df["Close_rolling_std_5"] = df["Close"].rolling(window=5, min_periods=1).std().fillna(0.0)

#     # diff and pct change (canonical names)
#     if "Close_diff" not in df.columns:
#         df["Close_diff"] = df["Close"].diff().fillna(0)
#     if "Close_pct_change" not in df.columns:
#         df["Close_pct_change"] = df["Close"].pct_change().fillna(0)

#     # === Define features ===
#     # Exclude explicit excluded columns and the target
#     feature_cols = [col for col in df.columns if col not in exclude_cols]

#     # Remove duplicates preserving order (guard against duplicated column names in CSV)
#     feature_cols = list(dict.fromkeys(feature_cols))

#     if not feature_cols:
#         raise ValueError("No feature columns found after excluding non-features.")

#     # === Scale features only ===
#     feature_scaler = MinMaxScaler(feature_range=(0, 1))
#     df[feature_cols] = feature_scaler.fit_transform(df[feature_cols].astype(float))

#     # Save feature scaler
#     with open(SAVED_MODELS_DIR / "tft_feature_scaler.pkl", "wb") as f:
#         pickle.dump(feature_scaler, f)

#     # === Train/Val Split ===
#     total = len(df)
#     split_idx = int(total * (1 - test_size))
#     val_size = total - split_idx
#     if val_size < min_val_samples:
#         print(f" Validation size too small ({val_size}), adjusting split to guarantee {min_val_samples} validation samples.")
#         split_idx = max(0, total - min_val_samples)

#     train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
#     val_df = df.iloc[max(0, split_idx - seq_len):].copy().reset_index(drop=True)

#     # === TimeSeriesDataSet ===
#     training = TimeSeriesDataSet(
#         train_df,
#         time_idx="time_idx",
#         target=target_col,
#         group_ids=["group_id"],
#         min_encoder_length=seq_len,
#         max_encoder_length=seq_len,
#         min_prediction_length=1,
#         max_prediction_length=1,
#         static_categoricals=[],
#         time_varying_known_reals=[],
#         time_varying_unknown_reals=feature_cols,
#         target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
#         add_relative_time_idx=True,
#         add_target_scales=True,
#         add_encoder_length=True,
#         allow_missing_timesteps=True,
#     )

#     validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

#     train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
#     val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

#     print(f" TFT Data prepared: Train={len(training)}, Val={len(validation)}, Features={len(feature_cols)}")
#     return training, validation, train_dataloader, val_dataloader, feature_cols


def prepare_tft_data(seq_len=30, test_size=0.2, debug=False, min_val_samples=120):
    """
    Prepare data for Temporal Fusion Transformer using the same scaled data
    and features as LSTM (via prepare_shared_data).
    Ensures both models are trained on identical inputs.
    """
    # === Get pre-scaled, unified dataset ===
    df, feature_cols = prepare_shared_data(seq_len=seq_len, is_training=True)
    df["date"] = pd.to_datetime(df["date"])
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

    target_col = "Close_scaled"  #  Already scaled in shared preprocessor
    exclude_cols = ["date", "ticker", "sentiment_label_mode", "time_idx", "group_id", target_col]

    # === Train/Validation Split ===
    total = len(df)
    split_idx = int(total * (1 - test_size))
    val_size = total - split_idx
    if val_size < min_val_samples:
        print(f" Validation size too small ({val_size}), adjusting to {min_val_samples}.")
        split_idx = max(0, total - min_val_samples)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[max(0, split_idx - seq_len):].copy().reset_index(drop=True)

    # === Build TFT Datasets ===
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["group_id"],
        min_encoder_length=seq_len,
        max_encoder_length=seq_len,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        time_varying_known_reals=[],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=None,  # Already scaled!
        add_relative_time_idx=True,
        add_target_scales=False,  # Disabled since we scaled manually
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    print(f"TFT Data prepared (shared): Train={len(training)}, Val={len(validation)}, Features={len(feature_cols)}")
    return training, validation, train_dataloader, val_dataloader, feature_cols






# === Utility: extract tensor from various model outputs ===
def extract_prediction_tensor(model_output):
    """
    Try to extract a plain torch.Tensor containing predictions from
    TemporalFusionTransformer output. Handles:
      - torch.Tensor
      - dict-like containing keys 'prediction' / 'pred' / 'predictions' / 'output'
      - objects with attributes 'prediction' / 'pred' / ...
      - indexable outputs like out[0]
    Returns: torch.Tensor
    Raises TypeError if unable to extract.
    """
    out = model_output
    # direct tensor
    if isinstance(out, torch.Tensor):
        return out
    # dict-like
    if isinstance(out, dict):
        for key in ("prediction", "pred", "predictions", "output"):
            if key in out and isinstance(out[key], torch.Tensor):
                return out[key]
        # fallback: first tensor-valued entry
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    # object with attribute
    for attr in ("prediction", "pred", "predictions", "output"):
        if hasattr(out, attr):
            tensor = getattr(out, attr)
            if isinstance(tensor, torch.Tensor):
                return tensor
    # indexable (namedtuple-like)
    try:
        candidate = out[0]
        if isinstance(candidate, torch.Tensor):
            return candidate
    except Exception:
        pass
    raise TypeError(f"Could not extract prediction tensor from model output of type {type(out)}")


# === TFT Lightning wrapper that calls forward and computes loss ourselves ===
class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, core_tft: TemporalFusionTransformer, lr=1e-3):
        super().__init__()
        self.tft = core_tft
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.tft(x)
        pred = extract_prediction_tensor(out)
        # ensure prediction is tensor with proper shape
        loss = self.tft.loss(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.tft(x)
        pred = extract_prediction_tensor(out)
        loss = self.tft.loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # simple scheduler: ReduceLROnPlateau via native PyTorch scheduler config in Lightning
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# === Train Function ===
def train_tft(
    seq_len=60,                      # longer lookback for better trend capture
    hidden_size=96,                  #  larger hidden layer
    attention_head_size=4,           #  more attention heads
    dropout=0.2,                     #  mild regularization
    batch_size=64,
    epochs=50,
    lr=0.001,
    debug=False
):
    """
    Train an enhanced Temporal Fusion Transformer model on CPU.
    Improvements:
      - Increased context window and model capacity for better trend learning
      - Uses MAE loss for robustness
      - CPU-safe training setup (no MPS)
    """

    # === Prepare Data ===
    training, validation, train_dataloader, val_dataloader, feature_cols = prepare_tft_data(
        seq_len=seq_len, test_size=0.2, debug=debug
    )

    # === Build TFT Core Model ===
    tft_core = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size,
        output_size=1,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # === Lightning Wrapper ===
    wrapped = TFTLightningWrapper(tft_core, lr=lr)

    # === Callbacks ===
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    # === Trainer (CPU-only) ===
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu",      #  ensures MPS/GPU won't be used
        devices=1,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # === Train Model ===
    trainer.fit(
        wrapped,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    print("\n✅ TFT Training complete.")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint path: {best_model_path}")

    # === Save Core Model & Dataset ===
    ensure_dir(SAVED_MODELS_DIR)
    final_model_path = SAVED_MODELS_DIR / "tft.pth"
    torch.save(
        {
            "model_state_dict": tft_core.state_dict(),
            "feature_cols": feature_cols,
        },
        final_model_path,
    )

    with open(SAVED_MODELS_DIR / "tft_training_dataset.pkl", "wb") as f:
        pickle.dump(training, f)

    print(f"Final model saved to: {final_model_path}")
    print(f"Training dataset saved to: {SAVED_MODELS_DIR / 'tft_training_dataset.pkl'}")

    return best_model_path


# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT model for stock prediction (aligned with LSTM preprocessing)")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--attention_head_size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("  Starting TFT training...")
    train_tft(**vars(args))
    print("  Training complete!")
