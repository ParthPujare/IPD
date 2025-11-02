# src/evaluation/plots_lstm.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "models" / "evaluation"
df = pd.read_csv(OUT / "lstm_val_predictions.csv")
# Plot actual vs predicted (first 200 points)
plt.figure(figsize=(12,5))
plt.plot(df["y_true"].values[:200], label="y_true")
plt.plot(df["y_pred"].values[:200], label="y_pred")
plt.legend(); plt.title("LSTM: Actual vs Predicted (first 200 points)")
plt.savefig(OUT / "lstm_pred_vs_true.png", dpi=150)
plt.close()

# Residuals
resid = df["y_true"] - df["y_pred"]
plt.figure(figsize=(8,4))
plt.hist(resid, bins=80)
plt.title("Residual histogram")
plt.savefig(OUT / "lstm_residual_hist.png", dpi=150)
plt.close()

# residuals over time
plt.figure(figsize=(12,4))
plt.plot(resid[:200])
plt.title("Residuals over time (first 200 points)")
plt.savefig(OUT / "lstm_residuals_time.png", dpi=150)
plt.close()

print("Saved plots to", OUT)
