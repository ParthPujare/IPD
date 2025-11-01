"""
Temporal Fusion Transformer (TFT) training script using pytorch-forecasting.
Trains a TFT model to predict next-day closing price with covariates.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import sys
import pickle
import os

# ✅ Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = "cpu"

# TensorBoard logger
tb_logger = TensorBoardLogger("lightning_logs", name="tft_runs")

if not hasattr(np, 'float'):
    np.float = float

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir, get_accelerator

FEATURES_DATA_PATH = get_project_root() / "data" / "features.csv"
CHECKPOINT_DIR = get_project_root() / "models" / "checkpoints"
SAVED_MODELS_DIR = get_project_root() / "models" / "saved_models"


def prepare_tft_data(debug=False):
    df = pd.read_csv(FEATURES_DATA_PATH)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    if debug:
        df = df.tail(500).reset_index(drop=True)

    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

    target = "Close"

    exclude_cols = ["date", "ticker", "sentiment_label_mode", "time_idx", "group_id", target]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    technical_cols = [col for col in feature_cols if any(
        x in col.lower() for x in ["sma", "ema", "rsi", "returns", "volatility", "volume", "lag", "close_lag"]
    )]
    observed_reals = list(set(technical_cols + [col for col in feature_cols if "sentiment" in col.lower()] + 
                              [col for col in feature_cols if col in ["Open", "High", "Low"]]))
    observed_reals = [col for col in observed_reals if col in df.columns]

    split_idx = int(len(df) * 0.8)
    train_df = df[df["time_idx"] < df.iloc[split_idx]["time_idx"]]
    val_df = df[df["time_idx"] >= df.iloc[split_idx]["time_idx"]]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target,
        group_ids=["group_id"],
        min_encoder_length=30,
        max_encoder_length=60,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        time_varying_known_reals=[],
        time_varying_unknown_reals=observed_reals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=True, stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    print(f"Training samples: {len(training)}, Validation samples: {len(validation)}")
    print(f"Observed covariates: {len(observed_reals)}")
    print(f"Feature columns: {observed_reals[:5]}...")

    return training, validation, train_dataloader, val_dataloader, observed_reals


# ✅ ADD THIS WRAPPER CLASS (Lightning 2.x compatibility)
class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _extract_pred_tensor(self, model_output):
        """
        Extract a plain prediction tensor from model_output.
        Handles:
          - torch.Tensor
          - dict-like (e.g. {'prediction': tensor})
          - object with attribute .prediction or .pred
          - namedtuple-like with .prediction
        """
        out = model_output
        # direct tensor
        if isinstance(out, torch.Tensor):
            return out
        # dict-like
        if isinstance(out, dict):
            for key in ("prediction", "pred", "predictions", "output"):
                if key in out:
                    return out[key]
            # fallback: take the first tensor-like value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
        # object with attribute
        for attr in ("prediction", "pred", "predictions", "output"):
            if hasattr(out, attr):
                tensor = getattr(out, attr)
                if isinstance(tensor, torch.Tensor):
                    return tensor
        # namedtuple / attribute access fallback
        try:
            # some outputs support indexing like out[0]
            candidate = out[0]
            if isinstance(candidate, torch.Tensor):
                return candidate
        except Exception:
            pass
        # if nothing worked, raise clear error
        raise TypeError(f"Could not extract prediction tensor from model output of type {type(out)}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        y_pred = self._extract_pred_tensor(out)
        # model.loss expects (y_pred, target)
        loss = self.model.loss(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        y_pred = self._extract_pred_tensor(out)
        loss = self.model.loss(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # try to use model's configured learning rate if present
        lr = getattr(self.model.hparams, "learning_rate", None)
        if lr is None:
            lr = 1e-3
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    

def train_tft(hidden_size=32, attention_head_size=1, dropout=0.1,
              batch_size=64, epochs=50, lr=0.03, debug=False):

    training, validation, train_dataloader, val_dataloader, feature_cols = prepare_tft_data(debug=debug)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size,
        output_size=1,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # ✅ Wrap TFT model for Lightning 2.x
    wrapped_model = TFTLightningWrapper(tft)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)

    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu",
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True
    )

    trainer.fit(wrapped_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    print("\nTraining complete. Checking best checkpoint...")

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not Path(best_model_path).is_file():
        checkpoint_dir = Path("/Users/parth/Documents/Work/Code/College/IPD/IPD/models/checkpoints")
        ckpts = list(checkpoint_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        best_model_path = sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]

    print(f"Best model checkpoint loaded from: {best_model_path}")

    ensure_dir(SAVED_MODELS_DIR)
    print(f"Loading best checkpoint manually from: {best_model_path}")

    # Recreate model with the same parameters as during training
    best_model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size,
        output_size=1,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # Load weights manually (since load_from_checkpoint fails)
    checkpoint = torch.load(best_model_path, map_location="cpu")
    best_model.load_state_dict(checkpoint["state_dict"], strict=False)
    final_model_path = SAVED_MODELS_DIR / "tft.pth"

    torch.save({
    "model_state_dict": best_model.state_dict(),
    "training_dataset": training,
    "feature_cols": feature_cols
    }, final_model_path)

    print(f"Final model saved successfully at: {final_model_path}")

    with open(SAVED_MODELS_DIR / "tft_training_dataset.pkl", "wb") as f:
        pickle.dump(training, f)

    print(f"Training dataset saved to {SAVED_MODELS_DIR / 'tft_training_dataset.pkl'}")

    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT model for stock prediction")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--attention_head_size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("Starting TFT training...")
    train_tft(**vars(args))
    print("Training complete!")
