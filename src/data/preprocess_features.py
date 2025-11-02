# src/data/preprocess_features.py

"""
Utility script to enhance features.csv with additional technical indicators.
Non-destructive: outputs a new file 'features_enhanced.csv'.
"""

import pandas as pd
from pathlib import Path
from src.utils.feature_utils import add_basic_technical_features

def main():
    data_dir = Path(__file__).parent.parent.parent / "data"
    input_file = data_dir / "features.csv"
    output_file = data_dir / "features_enhanced.csv"

    print(f" Reading original features: {input_file}")
    df = pd.read_csv(input_file)
    print(f" Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Add technical features
    print(" Adding engineered features...")
    df_enhanced = add_basic_technical_features(df)

    # ✅ Add short-term momentum and return features (new)
    print(" Adding short-term momentum features...")
    df_enhanced["return_1d"] = df_enhanced["Close"].pct_change()
    df_enhanced["return_3d"] = df_enhanced["Close"].pct_change(3)
    df_enhanced["momentum_3d"] = df_enhanced["Close"] - df_enhanced["Close"].shift(3)

    # Drop initial NaNs from these new features
    df_enhanced = df_enhanced.dropna().reset_index(drop=True)

    # Save to disk
    df_enhanced.to_csv(output_file, index=False)
    print(f" Saved enhanced features to: {output_file}")
    print(f"New shape: {df_enhanced.shape}")
    print(f"Sample columns added: {[c for c in df_enhanced.columns if 'lag' in c or 'sma' in c or 'return' in c][:10]}")

if __name__ == "__main__":
    main()
