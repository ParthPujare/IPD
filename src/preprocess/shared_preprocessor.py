import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from src.utils.helpers import get_project_root, ensure_dir

ROOT = get_project_root()
FEATURES_DATA_PATH = ROOT / "data" / "features_enhanced.csv"
SAVED_MODELS_DIR = ROOT / "models" / "saved_models"
ensure_dir(SAVED_MODELS_DIR)


def prepare_shared_data(seq_len=30, test_size=0.2, is_training=False):
    """
    Unified preprocessing pipeline for both LSTM and TFT.
    Ensures identical features and scaling are used across both models.
    If is_training=True, it fits new scalers and saves them to disk.
    If is_training=False, it loads the saved scalers and transforms the data.
    """
    df = pd.read_csv(FEATURES_DATA_PATH).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["group_id"] = df.get("ticker", "ADANIGREEN.NS").fillna("ADANIGREEN.NS")

    # --- Derived features (used by both models) ---
    df["Close_diff"] = df["Close"].diff().fillna(0)
    df["Close_pct_change"] = df["Close"].pct_change().fillna(0)
    df["Close_rolling_mean_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["Close_rolling_std_5"] = df["Close"].rolling(5, min_periods=1).std().fillna(0)

    # --- Define features ---
    exclude_cols = ["date", "ticker", "sentiment_label_mode", "Close", "time_idx", "group_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = list(dict.fromkeys(feature_cols))

    # --- Unified scaling ---
    if is_training:
        feature_scaler = MinMaxScaler((0, 1))
        target_scaler = StandardScaler()

        df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
        df["Close_scaled"] = target_scaler.fit_transform(df["Close"].values.reshape(-1, 1))

        # --- Save scalers ---
        with open(SAVED_MODELS_DIR / "shared_feature_scaler.pkl", "wb") as f:
            pickle.dump(feature_scaler, f)
        with open(SAVED_MODELS_DIR / "shared_target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)
        with open(SAVED_MODELS_DIR / "shared_features.pkl", "wb") as f:
            pickle.dump(feature_cols, f)
        print(f"FIT new scalers and prepared data — {len(feature_cols)} features scaled.")
    else:
        # Load existing scalers
        if not (SAVED_MODELS_DIR / "shared_feature_scaler.pkl").exists():
             raise FileNotFoundError("Scalers not found. Run training first.")
        with open(SAVED_MODELS_DIR / "shared_feature_scaler.pkl", "rb") as f:
            feature_scaler = pickle.load(f)
        with open(SAVED_MODELS_DIR / "shared_target_scaler.pkl", "rb") as f:
            target_scaler = pickle.load(f)
            
        # Add missing columns if any with 0
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        df[feature_cols] = feature_scaler.transform(df[feature_cols])
        df["Close_scaled"] = target_scaler.transform(df["Close"].values.reshape(-1, 1))
        print(f"TRANSFORMED using existing scalers — {len(feature_cols)} features scaled.")

    return df, feature_cols
