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
from pytorch_forecasting.metrics import MAE, RMSE, MAPE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import sys
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir, get_accelerator


FEATURES_DATA_PATH = get_project_root() / "data" / "features.csv"
CHECKPOINT_DIR = get_project_root() / "models" / "checkpoints"
SAVED_MODELS_DIR = get_project_root() / "models" / "saved_models"


def prepare_tft_data(debug=False):
    """
    Prepare data for TFT training in TimeSeriesDataSet format.
    
    Args:
        debug (bool): If True, use smaller dataset for debugging
    
    Returns:
        tuple: (training_dataset, validation_dataset, data_module)
    """
    # Load features
    df = pd.read_csv(FEATURES_DATA_PATH)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    
    if debug:
        # Use only last 500 rows for debugging
        df = df.tail(500).reset_index(drop=True)
    
    # Create time index (days since first date)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    
    # Set group (ticker) - single group for now
    df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")
    
    # Target variable
    target = "Close"
    
    # Known covariates (future values we know)
    known_reals = []
    
    # Observed covariates (past values we observe)
    observed_reals = []
    
    # Static covariates (constant per group)
    static_categoricals = []
    
    # Feature columns (exclude non-feature columns)
    exclude_cols = ["date", "ticker", "sentiment_label_mode", "time_idx", "group_id", target]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split observed and known
    # Technical indicators are observed (past)
    technical_cols = [col for col in feature_cols if any(
        x in col.lower() for x in ["sma", "ema", "rsi", "returns", "volatility", "volume", "lag", "close_lag"]
    )]
    observed_reals = technical_cols
    
    # Sentiment can be considered as observed (we have it for past dates)
    sentiment_cols = [col for col in feature_cols if "sentiment" in col.lower()]
    observed_reals.extend(sentiment_cols)
    
    # Price columns (Open, High, Low) are observed
    price_cols = [col for col in feature_cols if col in ["Open", "High", "Low"]]
    observed_reals.extend(price_cols)
    
    # Remove duplicates
    observed_reals = list(set(observed_reals))
    
    # If any remaining columns, add to observed
    remaining = [col for col in feature_cols if col not in observed_reals]
    observed_reals.extend(remaining)
    
    # Ensure all columns exist in dataframe
    observed_reals = [col for col in observed_reals if col in df.columns]
    
    # Validation split
    max_encoder_length = 60  # Look back 60 days
    max_prediction_length = 1  # Predict next day
    
    # Split point: use last 20% for validation
    split_idx = int(len(df) * 0.8)
    train_df = df[df["time_idx"] < df.iloc[split_idx]["time_idx"]]
    val_df = df[df["time_idx"] >= df.iloc[split_idx]["time_idx"]]
    
    # Training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target,
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=observed_reals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    # Validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=True,
        stop_randomization=True
    )
    
    # Create data loader
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    print(f"Training samples: {len(training)}, Validation samples: {len(validation)}")
    print(f"Observed covariates: {len(observed_reals)}")
    print(f"Feature columns: {observed_reals[:5]}...")  # Show first 5
    
    return training, validation, train_dataloader, val_dataloader, observed_reals


def train_tft(
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    batch_size=64,
    epochs=50,
    lr=0.03,
    debug=False
):
    """
    Train TFT model.
    
    Args:
        hidden_size (int): Hidden size for TFT
        attention_head_size (int): Attention head size
        dropout (float): Dropout rate
        batch_size (int): Batch size
        epochs (int): Number of epochs
        lr (float): Learning rate
        debug (bool): Debug mode (smaller dataset)
    """
    # Prepare data
    training, validation, train_dataloader, val_dataloader, feature_cols = prepare_tft_data(debug=debug)
    
    # Create TFT model
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
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )
    
    # Get accelerator
    accelerator = get_accelerator()
    print(f"Training on accelerator: {accelerator}")
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=0.1
    )
    
    # Train
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    
    # Load best model and save for inference
    ensure_dir(SAVED_MODELS_DIR)
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    # Save model and training dataset for inference
    final_model_path = SAVED_MODELS_DIR / "tft.pth"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "training_dataset": training,
        "feature_cols": feature_cols
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training dataset separately (needed for inference)
    training_dataset_path = SAVED_MODELS_DIR / "tft_training_dataset.pkl"
    with open(training_dataset_path, "wb") as f:
        pickle.dump(training, f)
    print(f"Training dataset saved to {training_dataset_path}")
    
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT model for stock prediction")
    parser.add_argument("--hidden_size", type=int, default=32, help="TFT hidden size")
    parser.add_argument("--attention_head_size", type=int, default=1, help="Attention head size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Debug mode (smaller dataset)")
    
    args = parser.parse_args()
    
    print("Starting TFT training...")
    train_tft(
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        debug=args.debug
    )
    print("Training complete!")

