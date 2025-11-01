"""
Streamlit dashboard for AdaniGreenPredictor.
Displays stock prices, news, sentiment, predictions, and LLM summaries.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root

# Lazy imports for prediction modules (only load when needed)
# This allows dashboard to run even if training dependencies aren't installed
def safe_import_predict():
    """Safely import prediction modules, return None if unavailable."""
    try:
        from src.inference.predict import predict_next_day
        return predict_next_day
    except ImportError as e:
        st.warning(f"Prediction module unavailable: {e}")
        return None

def safe_import_ensemble():
    """Safely import ensemble module, return None if unavailable."""
    try:
        from src.inference.ensemble import ensemble_predict
        return ensemble_predict
    except ImportError:
        return None

def safe_import_llm():
    """Safely import LLM module, return None if unavailable."""
    try:
        from src.llm.llm_summary import generate_summary
        return generate_summary
    except ImportError:
        return None


STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"
NEWS_DATA_PATH = get_project_root() / "data" / "news_data.csv"


def load_stock_data():
    """Load stock data from CSV."""
    if STOCK_DATA_PATH.exists():
        df = pd.read_csv(STOCK_DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def load_news_data():
    """Load news data from CSV."""
    if NEWS_DATA_PATH.exists():
        df = pd.read_csv(NEWS_DATA_PATH)
        if "published_date" in df.columns:
            df["published_date"] = pd.to_datetime(df["published_date"])
        return df
    return pd.DataFrame()


def get_sentiment_summary(news_df):
    """Get summary of sentiment from news data."""
    if news_df.empty or "sentiment_score" not in news_df.columns:
        return "neutral"
    
    mean_sentiment = news_df["sentiment_score"].mean()
    
    if mean_sentiment > 0.2:
        return "mostly positive"
    elif mean_sentiment < -0.2:
        return "mostly negative"
    else:
        return "neutral"


def plot_price_chart(df, predicted_price=None, model_name=None):
    """Create interactive price chart with prediction."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Stock Price (Close)", "Volume"),
        row_width=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["Close"],
            name="Close Price",
            line=dict(color="blue", width=2)
        ),
        row=1, col=1
    )
    
    # Add prediction point if available
    if predicted_price is not None:
        last_date = df["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        fig.add_trace(
            go.Scatter(
                x=[next_date],
                y=[predicted_price],
                name=f"Predicted ({model_name})",
                mode="markers",
                marker=dict(
                    color="red",
                    size=12,
                    symbol="star"
                )
            ),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["Volume"],
            name="Volume",
            marker_color="lightblue"
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text="Adani Green Energy Stock Price & Volume",
        hovermode="x unified"
    )
    
    return fig


def color_sentiment(val):
    """Color code sentiment labels."""
    if val == "positive":
        return "background-color: #90EE90"  # Light green
    elif val == "negative":
        return "background-color: #FFB6C1"  # Light pink
    else:
        return "background-color: #FFFFE0"  # Light yellow


# Page configuration
st.set_page_config(
    page_title="AdaniGreenPredictor",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 AdaniGreenPredictor Dashboard")
st.markdown("Real-time stock price prediction for Adani Green Energy (ADANIGREEN.NS)")

# Sidebar
st.sidebar.title("⚙️ Settings")

# Model selector
model_option = st.sidebar.selectbox(
    "Select Model",
    ["LSTM", "TFT", "Ensemble"],
    help="Choose prediction model"
)

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.info("Refreshing data... Please run main.py to update data files.")
    st.rerun()

# Load data
with st.spinner("Loading data..."):
    stock_df = load_stock_data()
    news_df = load_news_data()

# Main content
if stock_df.empty:
    st.error("No stock data available. Please run main.py to fetch data first.")
else:
    # Display stock data info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        last_price = stock_df["Close"].iloc[-1]
        st.metric("Last Price", f"₹{last_price:.2f}")
    
    with col2:
        price_change = stock_df["Close"].iloc[-1] - stock_df["Close"].iloc[-2] if len(stock_df) > 1 else 0
        st.metric("1-Day Change", f"₹{price_change:.2f}")
    
    with col3:
        date_range = f"{stock_df['date'].min().date()} to {stock_df['date'].max().date()}"
        st.metric("Data Range", f"{len(stock_df)} days")
    
    with col4:
        sentiment_summary = get_sentiment_summary(news_df) if not news_df.empty else "neutral"
        st.metric("News Sentiment", sentiment_summary)
    
    st.markdown("---")
    
    # Prediction section
    st.subheader("🎯 Price Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Lazy load prediction functions
        predict_next_day = safe_import_predict()
        ensemble_predict_func = safe_import_ensemble()
        
        if predict_next_day is None:
            st.error("Prediction modules unavailable. Please ensure all dependencies are installed.")
            st.info("Run: `pip install -r requirements.txt` in your virtual environment.")
            st.info("Also ensure you're running Streamlit from the venv: `venv\\Scripts\\python -m streamlit run app/dashboard.py`")
            predicted_price = None
            model_name_display = None
            prediction_result = None
        else:
            try:
                if model_option == "Ensemble":
                    if ensemble_predict_func is None:
                        st.error("Ensemble prediction unavailable. Please ensure all dependencies are installed.")
                        prediction_result = None
                    else:
                        prediction_result = ensemble_predict_func()
                        model_name_display = "Ensemble (LSTM + TFT)"
                else:
                    prediction_result = predict_next_day(model_option)
                    model_name_display = model_option
                
                if prediction_result:
                    predicted_price = prediction_result["predicted_price"]
                    predicted_change_pct = prediction_result["predicted_change_pct"]
                    confidence = prediction_result.get("confidence", 50)
                    
                    st.success(f"**Predicted Price: ₹{predicted_price:.2f}**")
                    st.info(f"Predicted Change: {predicted_change_pct:+.2f}% | Confidence: {confidence:.1f}%")
                else:
                    predicted_price = None
                    model_name_display = None
                    
            except FileNotFoundError as e:
                st.error(f"Model files not found: {e}")
                st.info("Please train models first by running training scripts on Mac M4.")
                predicted_price = None
                model_name_display = None
                prediction_result = None
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Please train models first by running training scripts on Mac M4.")
                predicted_price = None
                model_name_display = None
                prediction_result = None
    
    with col2:
        if prediction_result:
            change_direction = "📈" if predicted_change_pct > 0 else "📉"
            st.markdown(f"### {change_direction}")
            st.markdown(f"**{abs(predicted_change_pct):.2f}%** change predicted")
    
    # Price chart
    st.subheader("📊 Stock Price Chart")
    fig = plot_price_chart(stock_df, predicted_price, model_name_display)
    st.plotly_chart(fig, use_container_width=True)
    
    # LLM Summary - Always show (with or without predictions)
    st.subheader("🤖 AI Explanation")
    
    # Prepare context dict - works with or without predictions
    if prediction_result:
        context_dict = {
            "predicted_price": predicted_price,
            "last_price": prediction_result["last_price"],
            "predicted_change_pct": predicted_change_pct,
            "sentiment_summary": sentiment_summary,
            "model_name": model_name_display if model_name_display else "Historical Analysis"
        }
    else:
        # Create context from current stock data even without predictions
        current_price = stock_df["Close"].iloc[-1]
        prev_price = stock_df["Close"].iloc[-2] if len(stock_df) > 1 else current_price
        price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        context_dict = {
            "predicted_price": current_price,  # Use current as "predicted"
            "last_price": current_price,
            "predicted_change_pct": price_change_pct,
            "sentiment_summary": sentiment_summary,
            "model_name": "Current Market Analysis"
        }
    
    try:
        generate_summary_func = safe_import_llm()
        if generate_summary_func:
            with st.spinner("Generating AI insights..."):
                summary = generate_summary_func(context_dict)
                # Display with markdown formatting
                st.markdown(summary)
        else:
            st.warning("⚠️ LLM summary unavailable. Dependencies may not be installed.")
            st.info("💡 Run `pip install -r requirements.txt` to install all dependencies including transformers for LLM.")
    except Exception as e:
        st.warning(f"⚠️ Could not generate LLM summary: {e}")
        st.info("💡 The LLM summary feature uses flan-t5-small model. Ensure transformers library is installed.")
    
    # News section
    st.subheader("📰 Latest News & Sentiment")
    
    if news_df.empty:
        st.warning("No news data available. Please run main.py to fetch news.")
    else:
        # Display latest news with sentiment
        display_cols = ["published_date", "title", "sentiment_label", "sentiment_score", "source", "url"]
        available_cols = [col for col in display_cols if col in news_df.columns]
        
        if available_cols:
            news_display = news_df[available_cols].head(10).copy()
            
            # Style sentiment labels
            if "sentiment_label" in news_display.columns:
                styled_news = news_display.style.applymap(
                    color_sentiment,
                    subset=["sentiment_label"]
                )
                st.dataframe(styled_news, use_container_width=True, hide_index=True)
            else:
                st.dataframe(news_display, use_container_width=True, hide_index=True)
            
            # Sentiment distribution
            if "sentiment_score" in news_df.columns:
                st.subheader("📊 Sentiment Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_counts = news_df["sentiment_label"].value_counts()
                    fig_sentiment = go.Figure(data=[
                        go.Bar(x=sentiment_counts.index, y=sentiment_counts.values)
                    ])
                    fig_sentiment.update_layout(
                        title="Sentiment Label Counts",
                        xaxis_title="Sentiment",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    fig_score = go.Figure(data=[
                        go.Histogram(x=news_df["sentiment_score"], nbinsx=20)
                    ])
                    fig_score.update_layout(
                        title="Sentiment Score Distribution",
                        xaxis_title="Sentiment Score",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
    
    # Admin section
    st.markdown("---")
    st.subheader("🔧 Admin Tools")
    
    with st.expander("Train Models"):
        st.warning("⚠️ **Training requires high compute. Run training on Mac M4 with MPS accelerator.**")
        st.code("""
# On Mac M4:
python src/training/train_lstm.py --epochs 50 --batch_size 64
python src/training/train_tft.py --epochs 50 --batch_size 64
        """)
    
    with st.expander("Load Pretrained Models"):
        st.info("Models should be saved in `models/saved_models/` after training.")
        if st.button("Check Model Files"):
            model_dir = get_project_root() / "models" / "saved_models"
            if model_dir.exists():
                model_files = list(model_dir.glob("*"))
                if model_files:
                    st.success(f"Found {len(model_files)} model file(s)")
                    for f in model_files:
                        st.text(f"  - {f.name}")
                else:
                    st.warning("No model files found. Train models first.")
            else:
                st.warning("Model directory does not exist.")

# Footer
st.markdown("---")
st.markdown("**AdaniGreenPredictor** | Built with PyTorch Lightning, pytorch-forecasting, and Streamlit")

