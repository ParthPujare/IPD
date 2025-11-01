"""
Fetch latest news headlines for Adani Green Energy using feedparser (RSS feeds).
Fallback to manual Google News RSS if NewsAPI is not available.
"""

import feedparser
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, ensure_dir


NEWS_DATA_PATH = get_project_root() / "data" / "news_data.csv"
NUM_NEWS_ITEMS = 10


def fetch_news_google_rss(query="Adani Green Energy", num_items=20):
    """
    Fetch news from Google News RSS feed.
    
    Args:
        query (str): Search query
        num_items (int): Number of news items to fetch
    
    Returns:
        pd.DataFrame: DataFrame with news headlines and metadata
    """
    # Google News RSS URL
    rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(rss_url)
        
        news_items = []
        for entry in feed.entries[:num_items]:
            try:
                # Parse published date
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6]).date()
                elif hasattr(entry, "published"):
                    try:
                        pub_date = pd.to_datetime(entry.published).date()
                    except:
                        pub_date = datetime.now().date()
                else:
                    pub_date = datetime.now().date()
                
                news_item = {
                    "title": entry.get("title", ""),
                    "published_date": pub_date,
                    "url": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "fetch_date": datetime.now().date()
                }
                news_items.append(news_item)
            except Exception as e:
                print(f"Error parsing news entry: {e}")
                continue
        
        df = pd.DataFrame(news_items)
        print(f"Fetched {len(df)} news items from Google News RSS")
        return df
    
    except Exception as e:
        print(f"Error fetching news from RSS: {e}")
        return pd.DataFrame(columns=["title", "published_date", "url", "summary", "source", "fetch_date"])


def fetch_news_newsapi(query="Adani Green Energy", num_items=10):
    """
    Fetch news from NewsAPI (requires API key in .env).
    
    Args:
        query (str): Search query
        num_items (int): Number of news items to fetch
    
    Returns:
        pd.DataFrame: DataFrame with news headlines and metadata
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("NewsAPI key not found in .env, falling back to RSS feed")
        return None
    
    try:
        import requests
        
        # NewsAPI endpoint
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}&language=en&pageSize={num_items}"
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        news_items = []
        for article in data.get("articles", [])[:num_items]:
            try:
                pub_date = None
                if article.get("publishedAt"):
                    pub_date = pd.to_datetime(article["publishedAt"]).date()
                else:
                    pub_date = datetime.now().date()
                
                news_item = {
                    "title": article.get("title", ""),
                    "published_date": pub_date,
                    "url": article.get("url", ""),
                    "summary": article.get("description", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "fetch_date": datetime.now().date()
                }
                news_items.append(news_item)
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue
        
        df = pd.DataFrame(news_items)
        print(f"Fetched {len(df)} news items from NewsAPI")
        return df
    
    except ImportError:
        print("requests library not installed, falling back to RSS")
        return None
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
        return None


def update_news_data():
    """
    Update news data CSV by fetching latest headlines.
    Appends new items and deduplicates by title.
    """
    ensure_dir(NEWS_DATA_PATH.parent)
    
    print("Fetching latest news for Adani Green Energy...")
    
    # Try NewsAPI first, fallback to RSS
    df = fetch_news_newsapi(num_items=NUM_NEWS_ITEMS)
    if df is None or df.empty:
        df = fetch_news_google_rss(num_items=NUM_NEWS_ITEMS)
    
    if df.empty:
        print("Warning: No news items fetched")
        return pd.DataFrame(columns=["title", "published_date", "url", "summary", "source", "fetch_date"])
    
    # Load existing data if it exists
    if NEWS_DATA_PATH.exists():
        try:
            existing_df = pd.read_csv(NEWS_DATA_PATH)
            existing_df["published_date"] = pd.to_datetime(existing_df["published_date"]).dt.date
            existing_df["fetch_date"] = pd.to_datetime(existing_df["fetch_date"]).dt.date
            
            # Merge and deduplicate by title
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["title"], keep="last")
            combined_df = combined_df.sort_values("published_date", ascending=False).reset_index(drop=True)
            
            print(f"Updated news data: {len(existing_df)} -> {len(combined_df)} records")
        except Exception as e:
            print(f"Error reading existing news data, using new data only: {e}")
            combined_df = df
    else:
        combined_df = df
        print(f"Created new news data file with {len(combined_df)} records")
    
    # Save to CSV
    combined_df.to_csv(NEWS_DATA_PATH, index=False)
    print(f"News data saved to {NEWS_DATA_PATH}")
    
    return combined_df


if __name__ == "__main__":
    # Test the fetch functionality
    print("Testing news data fetch...")
    df = update_news_data()
    if not df.empty:
        print(f"\nSample data (first 3 rows):")
        print(df.head(3))
        print(f"\nData shape: {df.shape}")
    else:
        print("No news data fetched")

