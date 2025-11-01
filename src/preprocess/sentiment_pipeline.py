"""
Sentiment analysis pipeline using FinBERT model from Hugging Face.
Processes news headlines and computes sentiment scores.
"""

import pandas as pd
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root, get_device


MODEL_NAME = "ProsusAI/finbert"
NEWS_DATA_PATH = get_project_root() / "data" / "news_data.csv"


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial news.
    """
    
    def __init__(self):
        """Initialize FinBERT model and tokenizer."""
        self.device = get_device()
        print(f"Loading FinBERT model on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            print(f"FinBERT model loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            print("Falling back to basic sentiment scoring")
            self.model = None
            self.tokenizer = None
        
        # Label mapping for FinBERT
        self.label_map = {0: "positive", 1: "neutral", 2: "negative"}
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            tuple: (sentiment_label, sentiment_score) where score is -1 (negative) to 1 (positive)
        """
        if self.model is None or text is None or pd.isna(text):
            return "neutral", 0.0
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            label = self.label_map[predicted_class]
            
            # Convert to score: positive=1, neutral=0, negative=-1
            if label == "positive":
                score = confidence
            elif label == "negative":
                score = -confidence
            else:
                score = 0.0
            
            return label, score
        
        except Exception as e:
            print(f"Error predicting sentiment for text: {e}")
            return "neutral", 0.0
    
    def add_sentiment(self, news_df):
        """
        Add sentiment scores to news dataframe.
        
        Args:
            news_df (pd.DataFrame): DataFrame with 'title' and optionally 'summary' columns
        
        Returns:
            pd.DataFrame: DataFrame with added 'sentiment_label' and 'sentiment_score' columns
        """
        if news_df.empty:
            return news_df
        
        df = news_df.copy()
        
        # Combine title and summary for better sentiment analysis
        df["text_for_sentiment"] = df.apply(
            lambda row: f"{row.get('title', '')} {row.get('summary', '')}".strip(),
            axis=1
        )
        
        print("Computing sentiment scores...")
        results = []
        for idx, text in enumerate(df["text_for_sentiment"]):
            if idx % 5 == 0:
                print(f"Processing {idx+1}/{len(df)}...")
            label, score = self.predict_sentiment(text)
            results.append({"sentiment_label": label, "sentiment_score": score})
        
        df["sentiment_label"] = [r["sentiment_label"] for r in results]
        df["sentiment_score"] = [r["sentiment_score"] for r in results]
        
        # Drop temporary column
        df = df.drop(columns=["text_for_sentiment"], errors="ignore")
        
        print(f"Sentiment analysis complete. Score range: [{df['sentiment_score'].min():.2f}, {df['sentiment_score'].max():.2f}]")
        
        return df


def add_sentiment_to_news_data():
    """
    Load news data, compute sentiment, and save back to CSV.
    """
    news_path = NEWS_DATA_PATH
    
    if not news_path.exists():
        print(f"News data file not found at {news_path}")
        return pd.DataFrame()
    
    # Load news data
    df = pd.read_csv(news_path)
    if df.empty:
        print("No news data to process")
        return df
    
    # Convert date columns
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"])
    
    # Initialize analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Add sentiment
    df = analyzer.add_sentiment(df)
    
    # Save back to CSV
    df.to_csv(news_path, index=False)
    print(f"Sentiment-added news data saved to {news_path}")
    
    return df


if __name__ == "__main__":
    # Test sentiment analysis
    print("Testing sentiment analysis pipeline...")
    
    # Test with sample text
    analyzer = FinBERTSentimentAnalyzer()
    
    test_texts = [
        "Adani Green Energy reports strong quarterly earnings and expansion plans",
        "Market uncertainty continues as Adani Green shares fluctuate",
        "Adani Green Energy faces regulatory challenges and project delays"
    ]
    
    print("\nSample sentiment predictions:")
    for text in test_texts:
        label, score = analyzer.predict_sentiment(text)
        print(f"Text: {text[:50]}...")
        print(f"  Sentiment: {label} (score: {score:.3f})\n")
    
    # Test with actual news data if available
    if NEWS_DATA_PATH.exists():
        print("\nProcessing actual news data...")
        df = add_sentiment_to_news_data()
        if not df.empty:
            print(f"\nSample results:")
            print(df[["title", "sentiment_label", "sentiment_score"]].head())

