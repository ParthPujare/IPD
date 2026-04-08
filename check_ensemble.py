import sys
import os
import pandas as pd
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.inference.predict import predict_next_day

def run_full_portfolio_check():
    # The list of tickers based on your saved_models folder
    tickers = [
        "ADANIGREEN.NS", 
        "BEL.NS", 
        "HDFCBANK.NS", 
        "INDIGO.NS", 
        "VEDL.NS"
    ]
    
    results = []

    print("\n" + "="*60)
    print("🚀 STARTING FULL PORTFOLIO ENSEMBLE CHECK")
    print("="*60)

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # 1. Fetch Predictions
        lstm_res = predict_next_day("LSTM", ticker)
        tft_res = predict_next_day("TFT", ticker)
        ens_res = predict_next_day("Ensemble", ticker)

        if lstm_res and tft_res and ens_res:
            l_p = lstm_res['predicted_price']
            t_p = tft_res['predicted_price']
            e_p = ens_res['predicted_price']
            
            # 2. Verify 60/40 Math
            expected = (l_p * 0.6) + (t_p * 0.4)
            status = "✅ PASSED" if abs(e_p - expected) < 0.1 else "⚠️ MATH ERROR"
            
            results.append({
                "Ticker": ticker,
                "LSTM": round(l_p, 2),
                "TFT": round(t_p, 2),
                "Ensemble": round(e_p, 2),
                "Status": status
            })
            print(f"   {status}: LSTM(₹{l_p:.2f}) & TFT(₹{t_p:.2f}) -> Ens(₹{e_p:.2f})")
        else:
            print(f"   ❌ FAILED: Missing .pt files for {ticker}")
            results.append({"Ticker": ticker, "Status": "❌ FAILED"})

    # Display Summary Table
    print("\n" + "="*60)
    print("📊 PORTFOLIO SUMMARY REPORT")
    print("="*60)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")

if __name__ == "__main__":
    run_full_portfolio_check()