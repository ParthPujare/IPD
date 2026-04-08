import subprocess
import sys

def run_pipeline():
    print("--- Starting Data Update Pipeline ---")
    
    # 1. Run Stock Fetcher
    print("Updating Stock Prices...")
    subprocess.run([sys.executable, "-m", "src.fetch.fetch_stock"])
    
    # 2. Run News Fetcher
    print("Updating News Sentiment...")
    subprocess.run([sys.executable, "-m", "src.fetch.fetch_news"])
    
    print("--- Data Update Complete. Launching Dashboard... ---")
    
    # 3. Launch Streamlit
    subprocess.run(["streamlit", "run", "app/dashboard.py"])

if __name__ == "__main__":
    run_pipeline()
    