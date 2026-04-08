import feedparser
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- PROJECT PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[3] 
sys.path.insert(0, str(project_root))

try:
    from src.utils.helpers import get_project_root, ensure_dir
except ModuleNotFoundError:
    project_root = current_file.parents[2]
    sys.path.insert(0, str(project_root))
    from src.utils.helpers import get_project_root, ensure_dir

# --- CONFIGURATION ---
NEWS_DATA_PATH = get_project_root() / "data" / "news_data.csv"
NUM_NEWS_ITEMS = 10

COMPANY_QUERIES = {
    "ADANIGREEN.NS": "Adani Green Energy",
    "VEDL.NS": "Vedanta Limited",
    "HDFCBANK.NS": "HDFC Bank",
    "INDIGO.NS": "Interglobe Aviation IndiGo",
    "BEL.NS": "Bharat Electronics Limited BEL"
}

# --- INITIALIZE FINBERT ---
print("Attempting to load FinBERT... (This may take a moment on first run)")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    """Analyze text using FinBERT and return score and label."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT indices: 0: Positive, 1: Negative, 2: Neutral
        pos_prob = probs[0][0].item()
        neg_prob = probs[0][1].item()
        neu_prob = probs[0][2].item()
        
        if pos_prob > neg_prob and pos_prob > neu_prob:
            return pos_prob, "Positive"
        elif neg_prob > pos_prob and neg_prob > neu_prob:
            return -neg_prob, "Negative"
        else:
            return 0.0, "Neutral"
    except Exception:
        return 0.0, "Neutral"

def fetch_news_google_rss(query, ticker, num_items=10):
    """Fetch news via RSS and apply sentiment analysis."""
    search_query = f"{query} stock market"
    rss_url = f"https://news.google.com/rss/search?q={search_query.replace(' ', '+')}&hl=en&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(rss_url)
        news_items = []
        for entry in feed.entries[:num_items]:
            title = entry.get("title", "")
            score, label = analyze_sentiment(title)
            
            news_items.append({
                "title": title,
                "published_date": datetime.now().strftime("%Y-%m-%d"),
                "url": entry.get("link", ""),
                "source": entry.get("source", {}).get("title", "Unknown"),
                "ticker": ticker,
                "sentiment_score": float(score),
                "sentiment_label": str(label),
                "fetch_date": datetime.now().strftime("%Y-%m-%d")
            })
        return pd.DataFrame(news_items)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

def update_news_data():
    """Main execution to update the master news CSV."""
    ensure_dir(NEWS_DATA_PATH.parent)
    all_news_dfs = []
    
    for ticker, company_name in COMPANY_QUERIES.items():
        print(f"Analyzing news for {company_name}...")
        df = fetch_news_google_rss(company_name, ticker, NUM_NEWS_ITEMS)
        if not df.empty:
            all_news_dfs.append(df)
        time.sleep(1)

    if not all_news_dfs:
        print("No data fetched.")
        return

    new_combined_df = pd.concat(all_news_dfs, ignore_index=True)

    if NEWS_DATA_PATH.exists():
        existing_df = pd.read_csv(NEWS_DATA_PATH)
        # Fix historical column name mismatches
        rename_map = {'sentiment_sentiment_score': 'sentiment_score', 'sentiment_score.1': 'sentiment_score'}
        existing_df = existing_df.rename(columns=rename_map)
        
        final_df = pd.concat([existing_df, new_combined_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["title", "ticker"], keep="last")
    else:
        final_df = new_combined_df

    final_df.to_csv(NEWS_DATA_PATH, index=False)
    print(f"✅ Successfully updated {NEWS_DATA_PATH}")

if __name__ == "__main__":
    update_news_data()