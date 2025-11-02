"""
LSTM training script using PyTorch Lightning.
Trains a regression model to predict next-day closing price.
"""

from sklearn.preprocessing import MinMaxScaler
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import torch.optim as optim
import sys
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir, get_accelerator


FEATURES_DATA_PATH = get_project_root() / "data" / "features_enhanced.csv"
CHECKPOINT_DIR = get_project_root() / "models" / "checkpoints"
SAVED_MODELS_DIR = get_project_root() / "models" / "saved_models"
SCALER_PATH = get_project_root() / "models" / "saved_models" / "lstm_scaler.pkl"


class StockDataset(Dataset):
    """Dataset for stock price sequences."""
    
    def __init__(self, sequences, targets):
        """
        Args:
            sequences (np.ndarray): Input sequences (N, seq_len, features)
            targets (np.ndarray): Target values (N,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


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

        # --- Layer Normalization ---
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # --- Fully Connected Layers ---
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
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # (batch, hidden*2)
        norm_out = self.layer_norm(last_output)
        output = self.fc(norm_out)
        return output

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

        # --- OneCycleLR scheduler ---
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
                "interval": "step",   # update every batch
                "frequency": 1
            }
        }



from sklearn.preprocessing import MinMaxScaler

def prepare_data(seq_len=30, test_size=0.2, debug=False):
    """
    Prepare data for LSTM training with both feature and target scaling.

    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_scaler, target_scaler, feature_cols)
    """
    df = pd.read_csv(FEATURES_DATA_PATH).sort_values("date").reset_index(drop=True)

    exclude_cols = ["date", "ticker", "sentiment_label_mode"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and c != "Close"]

    X = df[feature_cols].values
    y = df["Close"].values.reshape(-1, 1)

    if debug:
        X = X[-500:]
        y = y[-500:]

    # === Initialize scalers ===
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit & transform both
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    # === Create sequences ===
    sequences, targets = [], []
    for i in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[i - seq_len:i])
        targets.append(y_scaled[i])

    sequences = np.array(sequences)
    targets = np.array(targets).squeeze()

    # === Train/val split ===
    split_idx = int(len(sequences) * (1 - test_size))
    X_train, X_val = sequences[:split_idx], sequences[split_idx:]
    y_train, y_val = targets[:split_idx], targets[split_idx:]

    # === Save both scalers ===
    ensure_dir(SCALER_PATH.parent)
    feature_scaler_path = SCALER_PATH.parent / "lstm_feature_scaler.pkl"
    target_scaler_path = SCALER_PATH.parent / "lstm_target_scaler.pkl"

    with open(feature_scaler_path, "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(target_scaler_path, "wb") as f:
        pickle.dump(target_scaler, f)

    print(" Features and targets scaled using MinMaxScaler (0–1 range).")
    print(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Features={len(feature_cols)}")

    return X_train, X_val, y_train, y_val, feature_scaler, target_scaler, feature_cols



def train_lstm(
    seq_len=30,
    hidden_size=256,      # Increased model capacity for 60+ features
    num_layers=3,         # Deeper LSTM stack for richer temporal patterns
    dropout=0.3,          # More regularization to reduce overfitting
    batch_size=32,
    epochs=80,            # Longer training to ensure convergence
    lr=0.0005,            # Smaller learning rate for stable updates
    debug=False
):
    """
    Train LSTM model with improved configuration and save all metadata.
    """

    print("📊 Preparing data...")
    X_train, X_val, y_train, y_val, feature_scaler, target_scaler, feature_cols = prepare_data(
        seq_len=seq_len, debug=debug
    )

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"🧠 Building model: input={len(feature_cols)}, hidden={hidden_size}, layers={num_layers}")
    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f"lstm-h{hidden_size}-l{num_layers}-{{epoch:02d}}-{{val_loss:.4f}}",
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

    accelerator = get_accelerator()
    print(f"🚀 Training on accelerator: {accelerator}")

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    print(f" Best model saved at: {best_model_path}")

    ensure_dir(SAVED_MODELS_DIR)
    final_model_path = SAVED_MODELS_DIR / "lstm.pt"
    best_model = LSTMModel.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), final_model_path)
    print(f" Final model saved to {final_model_path}")

    # === Save scalers & feature metadata for inference ===
    with open(SAVED_MODELS_DIR / "lstm_feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(SAVED_MODELS_DIR / "lstm_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)
    with open(SAVED_MODELS_DIR / "lstm_features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(" Saved feature & target scalers and metadata.")
    print(" Training complete.")

    return best_model_path




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Debug mode (smaller dataset)")
    
    args = parser.parse_args()
    
    print("Starting LSTM training...")
    train_lstm(
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        debug=args.debug
    )
    print("Training complete!")

