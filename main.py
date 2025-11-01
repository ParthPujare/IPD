"""
Main entrypoint for AdaniGreenPredictor.
Refreshes data and launches Streamlit dashboard.
"""

import os
import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.fetch.fetch_stock import update_stock_data
from src.fetch.fetch_news import update_news_data
from src.preprocess.sentiment_pipeline import add_sentiment_to_news_data
from src.preprocess.features import build_features


def main():
    """
    Main function: refresh data and launch dashboard.
    """
    print("=" * 60)
    print("AdaniGreenPredictor - Starting Data Refresh")
    print("=" * 60)
    
    # Step 1: Fetch stock data
    print("\n[1/4] Fetching stock data...")
    try:
        stock_df = update_stock_data()
        print(f"✓ Stock data updated: {len(stock_df)} records")
    except Exception as e:
        print(f"✗ Error fetching stock data: {e}")
        print("Continuing with existing data...")
    
    # Step 2: Fetch news data
    print("\n[2/4] Fetching news data...")
    try:
        news_df = update_news_data()
        print(f"✓ News data updated: {len(news_df)} records")
    except Exception as e:
        print(f"✗ Error fetching news data: {e}")
        print("Continuing with existing data...")
        news_df = None
    
    # Step 3: Compute sentiment
    print("\n[3/4] Computing sentiment scores...")
    try:
        if news_df is not None and not news_df.empty:
            news_df = add_sentiment_to_news_data()
            print("✓ Sentiment analysis complete")
        else:
            print("⚠ Skipping sentiment analysis (no news data)")
    except Exception as e:
        print(f"✗ Error computing sentiment: {e}")
        print("Continuing without sentiment...")
    
    # Step 4: Build features
    print("\n[4/4] Building features...")
    try:
        features_df = build_features()
        print(f"✓ Features built: {len(features_df)} rows, {len(features_df.columns)} columns")
    except Exception as e:
        print(f"✗ Error building features: {e}")
        print("⚠ Continuing without updated features...")
    
    print("\n" + "=" * 60)
    print("Data refresh complete! Launching Streamlit dashboard...")
    print("=" * 60)
    
    # Launch Streamlit using venv Python explicitly
    dashboard_path = project_root / "app" / "dashboard.py"
    
    # Use venv Python to ensure correct interpreter
    import sys
    python_exe = sys.executable  # This will be venv Python if run from venv
    streamlit_cmd = [python_exe, "-m", "streamlit", "run", str(dashboard_path)]
    
    print(f"\nRunning: {' '.join(streamlit_cmd)}\n")
    print(f"Using Python: {python_exe}\n")
    
    try:
        subprocess.run(streamlit_cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")
    except Exception as e:
        print(f"\n✗ Error launching dashboard: {e}")
        print("\nYou can run the dashboard manually with:")
        print(f"  streamlit run {dashboard_path}")


if __name__ == "__main__":
    main()

