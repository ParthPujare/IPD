"""
LSTM training script for Multi-Stock Predictor.
Trains a model for a specific ticker using shared preprocessing.
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

# === Project imports ===
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir, get_accelerator
from src.preprocess.shared_preprocessor import prepare_shared_data

# === Path setup ===
ROOT = get_project_root()
SAVED_MODELS_DIR = ROOT / "models" / "saved_models"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
ensure_dir(SAVED_MODELS_DIR)
ensure_dir(CHECKPOINT_DIR)

class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.targets[idx]

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.criterion = nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        norm_out = self.layer_norm(lstm_out[:, -1, :])
        return self.fc(norm_out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def train_ticker(ticker, epochs=30, seq_len=30):
    print(f"\n--- 🏋️ Training Started for {ticker} ---")
    
    # 1. Get preprocessed data (this fits scalers)
    df_full, feature_cols = prepare_shared_data(seq_len=seq_len, is_training=True)
    
    # 2. Filter for specific ticker
    df = df_full[df_full['ticker'] == ticker].copy()
    
    if df.empty:
        print(f"❌ Error: No data found for {ticker}")
        return

    # 3. Create sequences
    X = df[feature_cols].values
    y = df["Close_scaled"].values
    
    sequences, targets = [], []
    for i in range(seq_len, len(X)):
        sequences.append(X[i-seq_len:i])
        targets.append(y[i])
    
    X_seq, y_target = np.array(sequences), np.array(targets)
    
    # 4. Split and Load
    split = int(len(X_seq) * 0.8)
    train_loader = DataLoader(StockDataset(X_seq[:split], y_target[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(StockDataset(X_seq[split:], y_target[split:]), batch_size=32)

    # 5. Train
    model = LSTMModel(input_size=len(feature_cols))
    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", devices=1, enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)

    # 6. Save unique model and metadata
    model_name = f"lstm_{ticker}.pt"
    torch.save(model.state_dict(), SAVED_MODELS_DIR / model_name)
    
    # Save the specific feature list used for this training
    with open(SAVED_MODELS_DIR / f"features_{ticker}.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
        
    print(f"✅ Success! Saved {model_name} to {SAVED_MODELS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    train_ticker(ticker=args.ticker, epochs=args.epochs)