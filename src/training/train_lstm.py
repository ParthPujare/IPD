"""
LSTM training script using PyTorch Lightning.
Trains a regression model to predict next-day closing price.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler

# === Project imports ===
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir, get_accelerator
from src.preprocess.shared_preprocessor import prepare_shared_data


# === Path setup ===
FEATURES_DATA_PATH = get_project_root() / "data" / "features_enhanced.csv"
CHECKPOINT_DIR = get_project_root() / "models" / "checkpoints"
SAVED_MODELS_DIR = get_project_root() / "models" / "saved_models"
ensure_dir(CHECKPOINT_DIR)
ensure_dir(SAVED_MODELS_DIR)


# === Dataset Class ===
class StockDataset(Dataset):
    """Dataset for stock price sequences."""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# === Model Definition ===
class LSTMModel(pl.LightningModule):
    """Bidirectional LSTM with LayerNorm and OneCycleLR for stock prediction."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # --- Normalization & Fully Connected ---
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        norm_out = self.layer_norm(last_output)
        return self.fc(norm_out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            anneal_strategy="cos",
            pct_start=0.3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


# === Training Function ===
def train_lstm(
    seq_len=30,
    hidden_size=256,
    num_layers=3,
    dropout=0.3,
    batch_size=32,
    epochs=80,
    lr=0.0005,
    debug=False
):
    """
    Train LSTM model using the shared preprocessing (same as TFT).
    """

    print(" Preparing shared data (unified for LSTM + TFT)...")
    df, feature_cols = prepare_shared_data(seq_len=seq_len, is_training=True)

    # --- Load the same shared scalers used for TFT ---
    with open(SAVED_MODELS_DIR / "shared_feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(SAVED_MODELS_DIR / "shared_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    # --- Create sequences ---
    X = df[feature_cols].values
    y = df["Close_scaled"].values.reshape(-1, 1)

    sequences, targets = [], []
    for i in range(seq_len, len(X)):
        sequences.append(X[i - seq_len:i])
        targets.append(y[i])

    sequences = np.array(sequences)
    targets = np.array(targets).squeeze()

    # --- Train/Val Split ---
    split_idx = int(len(sequences) * 0.8)
    X_train, X_val = sequences[:split_idx], sequences[split_idx:]
    y_train, y_val = targets[:split_idx], targets[split_idx:]

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # --- Build model ---
    print(f" Building model: input={len(feature_cols)}, hidden={hidden_size}, layers={num_layers}")
    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr
    )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f"lstm-h{hidden_size}-l{num_layers}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)

    # --- Trainer ---
    accelerator = get_accelerator()
    print(f" Training on accelerator: {accelerator}")
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # --- Train ---
    trainer.fit(model, train_loader, val_loader)

    # --- Save best model ---
    best_model_path = checkpoint_callback.best_model_path
    print(f" Best model saved at: {best_model_path}")

    ensure_dir(SAVED_MODELS_DIR)
    final_model_path = SAVED_MODELS_DIR / "lstm.pt"
    best_model = LSTMModel.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), final_model_path)
    print(f" Final model saved to {final_model_path}")

    # --- Save scalers and metadata ---
    with open(SAVED_MODELS_DIR / "lstm_feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(SAVED_MODELS_DIR / "lstm_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)
    with open(SAVED_MODELS_DIR / "lstm_features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(" Saved feature & target scalers and metadata.")
    print(" Training complete.")
    return best_model_path


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    print("Starting LSTM training...")
    train_lstm(**vars(args))
    print("Training complete!")
