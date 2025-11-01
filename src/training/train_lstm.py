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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    """LSTM model for stock price prediction."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, lr=0.001):
        """
        Args:
            input_size (int): Number of input features
            hidden_size (int): LSTM hidden dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            lr (float): Learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self.lr = lr
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, features)
        
        Returns:
            torch.Tensor: Predicted values (batch, 1)
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
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
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


def prepare_data(seq_len=30, test_size=0.2, debug=False):
    """
    Prepare data for LSTM training.
    
    Args:
        seq_len (int): Sequence length for LSTM
        test_size (float): Test split ratio
        debug (bool): If True, use smaller dataset for debugging
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler, feature_cols)
    """
    # Load features
    df = pd.read_csv(FEATURES_DATA_PATH)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter to numeric columns only (exclude date, ticker, sentiment_label_mode)
    exclude_cols = ["date", "ticker", "sentiment_label_mode"]
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != "Close"]
    
    # Use Close price as target
    target_col = "Close"
    
    # Select features
    X = df[feature_cols].values
    y = df[target_col].values
    
    if debug:
        # Use only last 500 rows for debugging
        X = X[-500:]
        y = y[-500:]
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(seq_len, len(X)):
        sequences.append(X[i - seq_len:i])
        targets.append(y[i])
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Split train/val
    split_idx = int(len(sequences) * (1 - test_size))
    X_train = sequences[:split_idx]
    y_train = targets[:split_idx]
    X_val = sequences[split_idx:]
    y_val = targets[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    # Save scaler
    ensure_dir(SCALER_PATH.parent)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Features={len(feature_cols)}")
    
    return X_train_scaled, X_val_scaled, y_train, y_val, scaler, feature_cols


def train_lstm(
    seq_len=30,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    batch_size=32,
    epochs=50,
    lr=0.001,
    debug=False
):
    """
    Train LSTM model.
    
    Args:
        seq_len (int): Sequence length
        hidden_size (int): LSTM hidden size
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        batch_size (int): Batch size
        epochs (int): Number of epochs
        lr (float): Learning rate
        debug (bool): Debug mode (smaller dataset)
    """
    # Prepare data
    X_train, X_val, y_train, y_val, scaler, feature_cols = prepare_data(
        seq_len=seq_len, debug=debug
    )
    
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="lstm-{epoch:02d}-{val_loss:.4f}",
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
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    
    # Save final model for inference
    ensure_dir(SAVED_MODELS_DIR)
    final_model_path = SAVED_MODELS_DIR / "lstm.pt"
    best_model = LSTMModel.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save feature columns for inference
    with open(SAVED_MODELS_DIR / "lstm_features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
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

