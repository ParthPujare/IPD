"""
Multi-Stock TFT training script using pytorch-forecasting.
Configuration: 64 Units, 2 Heads, AdamW, MAE Loss.
Trains a global model across all tickers.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import pytorch_lightning as pl
from src.preprocess.shared_preprocessor import prepare_shared_data
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import sys
import os

# Environment setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# === Path setup ===
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir

ROOT = get_project_root()
FEATURES_DATA_PATH = ROOT / "data" / "features_enhanced.csv"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
SAVED_MODELS_DIR = ROOT / "models" / "saved_models"
ensure_dir(CHECKPOINT_DIR)
ensure_dir(SAVED_MODELS_DIR)

def prepare_tft_data(seq_len=30, test_size=0.2, min_val_samples=120):
    """Prepare global data for TFT across all tickers."""
    # Use the shared preprocessor to get scaled data
    df, feature_cols = prepare_shared_data(seq_len=seq_len, is_training=True)
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # TFT Requirement: integer index per group
    df["time_idx"] = df.groupby("ticker").cumcount()
    df["ticker"] = df["ticker"].astype(str)

    target_col = "Close_scaled"
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - int(max_time_idx * test_size)

    # Build Training Dataset
    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["ticker"], 
        min_encoder_length=seq_len,
        max_encoder_length=seq_len,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["ticker"],
        time_varying_known_reals=[],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=None, 
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=False, stop_randomization=True)

    # Naming these 'train_loader' to match the function calls below
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    print(f"✅ Global TFT Data prepared: Tickers={df['ticker'].unique()}, Features={len(feature_cols)}")
    return training, validation, train_loader, val_loader, feature_cols

class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, core_tft: TemporalFusionTransformer, lr=0.001):
        super().__init__()
        self.tft = core_tft
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.tft(x)
        pred = out.prediction if hasattr(out, 'prediction') else out[0]
        loss = self.tft.loss(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.tft(x)
        pred = out.prediction if hasattr(out, 'prediction') else out[0]
        loss = self.tft.loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }

def train_tft(seq_len=30, hidden_size=64, attention_head_size=2, dropout=0.1, batch_size=64, epochs=50, lr=0.001):
    # 1. Prepare Global Data
    training, validation, train_loader, val_loader, feature_cols = prepare_tft_data(seq_len=seq_len)

    # 2. Build Model per your Specs
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
    )

    wrapped = TFTLightningWrapper(tft_core, lr=lr)

    # 3. Callbacks & Trainer
    ckpt = ModelCheckpoint(dirpath=CHECKPOINT_DIR, filename="tft_global-{epoch:02d}", monitor="val_loss", mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        callbacks=[ckpt, early_stop],
        logger=TensorBoardLogger("lightning_logs", name="tft_multi_stock")
    )

    # 4. Train - Fixed indentation and variable names
    trainer.fit(wrapped, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 5. Save Global Model
    ensure_dir(SAVED_MODELS_DIR)
    torch.save(tft_core.state_dict(), SAVED_MODELS_DIR / "tft_global.pt")
    
    with open(SAVED_MODELS_DIR / "tft_training_dataset.pkl", "wb") as f:
        pickle.dump(training, f)

    print(f"✅ Global TFT Model saved to {SAVED_MODELS_DIR / 'tft_global.pt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_tft(epochs=args.epochs)