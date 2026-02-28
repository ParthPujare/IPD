"""
Streamlit dashboard for AdaniGreenPredictor.
Displays stock prices, news, sentiment, predictions, and LLM summaries.
"""

import sys
from pathlib import Path

# --- Ensure correct project root and Python path ---
project_root = Path(__file__).resolve().parents[1]  # This points to /IPD
sys.path.insert(0, str(project_root))

#  Use your updated fetch functions
from src.fetch.fetch_stock import update_features_enhanced
from src.fetch.fetch_news import update_news_data

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

# Lazy imports for prediction modules
def safe_import_predict():
    try:
        from src.inference.predict import predict_next_day
        return predict_next_day
    except ImportError as e:
        st.warning(f"Prediction module unavailable: {e}")
        return None

def safe_import_ensemble():
    try:
        from src.inference.ensemble import ensemble_predict
        return ensemble_predict
    except ImportError:
        return None

def safe_import_llm():
    try:
        from src.llm.llm_summary import generate_summary
        return generate_summary
    except ImportError:
        return None


STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"
NEWS_DATA_PATH = get_project_root() / "data" / "news_data.csv"


def load_stock_data():
    """Load stock data from CSV (case-tolerant for 'Close' column)."""
    STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"
    if STOCK_DATA_PATH.exists():
        df = pd.read_csv(STOCK_DATA_PATH)
        # unify date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
        else:
            # no date column -> return empty
            return pd.DataFrame()
        # find 'Close' regardless of case
        if "Close" not in df.columns and "close" in df.columns:
            df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        else:
            # coerce existing Close to numeric
            if "Close" in df.columns:
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        # fill NaN last values conservatively (do not overwrite real zeros)
        df["Close"] = df["Close"].fillna(method="ffill").fillna(method="bfill")
        return df
    return pd.DataFrame()


def load_news_data():
    if NEWS_DATA_PATH.exists():
        df = pd.read_csv(NEWS_DATA_PATH)
        if "published_date" in df.columns:
            df["published_date"] = pd.to_datetime(df["published_date"])
        return df
    return pd.DataFrame()

def get_sentiment_summary(news_df):
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
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Stock Price (Close)", "Volume"),
        row_width=[0.7, 0.3]
    )

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["Close"], name="Close Price",
        line=dict(color="blue", width=2)
    ), row=1, col=1)

    if predicted_price is not None:
        last_date = df["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[next_date], y=[predicted_price],
            name=f"Predicted ({model_name})",
            mode="markers",
            marker=dict(color="red", size=12, symbol="star")
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["date"], y=df["Volume"], name="Volume",
        marker_color="lightblue"
    ), row=2, col=1)

    fig.update_layout(
        height=600,
        title_text="Adani Green Energy Stock Price & Volume",
        hovermode="x unified"
    )
    return fig

def color_sentiment(val):
    if val == "positive":
        return "background-color: #90EE90"
    elif val == "negative":
        return "background-color: #FFB6C1"
    else:
        return "background-color: #FFFFE0"


# --- PAGE CONFIG ---
st.set_page_config(page_title="AdaniGreenPredictor", page_icon="", layout="wide")
st.title(" AdaniGreenPredictor Dashboard")
st.markdown("Real-time stock price prediction for Adani Green Energy (ADANIGREEN.NS)")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")
model_option = st.sidebar.selectbox("Select Model", ["LSTM", "TFT", "Ensemble"], help="Choose prediction model")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 About the Models")
if model_option == "LSTM":
    st.sidebar.info("**LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) excellent at learning from sequences of data (like historical prices) to predict the future. It 'remembers' important past trends and 'forgets' noise.")
elif model_option == "TFT":
    st.sidebar.info("**TFT (Temporal Fusion Transformer)**: An advanced attention-based architecture designed specifically for time series. It can learn complex relationships and incorporate multiple features (like volume and sentiment) effectively.")
elif model_option == "Ensemble":
    st.sidebar.info("**Ensemble**: Combines predictions from both LSTM and TFT to provide a more robust and stable estimate, reducing the risk of a single model's error.")

# --- 🔧 AUTO DATA UPDATE ---
if st.sidebar.button(" Refresh Data (Manual)"):
    with st.spinner("Updating stock and news data..."):
        update_features_enhanced()
        update_news_data()
    st.success(" Data refreshed successfully!")
    st.rerun()

# --- 🔧 AUTO REFRESH BEFORE PREDICTION ---
with st.spinner(" Updating data before prediction..."):
    update_features_enhanced()
    update_news_data()

# --- LOAD DATA ---
with st.spinner("Loading data..."):
    stock_df = load_stock_data()
    news_df = load_news_data()

if stock_df.empty:
    st.error("No stock data available. Please run data fetcher first.")
else:
    # --- DASHBOARD METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Last Price", f"₹{stock_df['Close'].iloc[-1]:.2f}")
    with col2:
        change = stock_df["Close"].iloc[-1] - stock_df["Close"].iloc[-2] if len(stock_df) > 1 else 0
        st.metric("1-Day Change", f"₹{change:.2f}")
    with col3:
        st.metric("Data Range", f"{len(stock_df)} days")
    with col4:
        sentiment_summary = get_sentiment_summary(news_df) if not news_df.empty else "neutral"
        st.metric("News Sentiment", sentiment_summary)

    st.markdown("---")
    st.subheader("🎯 Price Prediction")

    col1, col2 = st.columns([2, 1])
    with col1:
        predict_next_day = safe_import_predict()
        ensemble_predict_func = safe_import_ensemble()

        if predict_next_day is None:
            st.error("Prediction modules unavailable.")
            predicted_price, model_name_display, prediction_result = None, None, None
        else:
            try:
                if model_option == "Ensemble":
                    if ensemble_predict_func is None:
                        st.error("Ensemble unavailable.")
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
                    predicted_price, model_name_display = None, None

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                predicted_price, model_name_display, prediction_result = None, None, None

    with col2:
        if prediction_result:
            icon = "📈" if predicted_change_pct > 0 else "📉"
            st.markdown(f"### {icon}")
            st.markdown(f"**{abs(predicted_change_pct):.2f}%** change predicted")

    # --- CHART ---
    st.subheader("📊 Stock Price Chart")
    fig = plot_price_chart(stock_df, predicted_price, model_name_display)
    st.plotly_chart(fig, use_container_width=True)

    # --- LLM Summary ---
    st.subheader("🤖 AI Explanation")
    generate_summary_func = safe_import_llm()
    context_dict = {
        "predicted_price": prediction_result["predicted_price"] if prediction_result else stock_df["Close"].iloc[-1],
        "last_price": stock_df["Close"].iloc[-1],
        "predicted_change_pct": prediction_result["predicted_change_pct"] if prediction_result else 0,
        "sentiment_summary": sentiment_summary,
        "model_name": model_name_display if prediction_result else "Current Market"
    }

    if generate_summary_func:
        with st.spinner("Generating AI insights..."):
            summary = generate_summary_func(context_dict)
            st.markdown(summary)
    else:
        st.info("LLM summary unavailable.")

    # --- NEWS SECTION ---
    st.subheader("📰 Latest News & Sentiment")
    if news_df.empty:
        st.warning("No news data available.")
    else:
        display_cols = ["published_date", "title", "sentiment_label", "sentiment_score", "source", "url"]
        available_cols = [c for c in display_cols if c in news_df.columns]
        news_display = news_df[available_cols].head(10).copy()
        if "sentiment_label" in news_display.columns:
            styled_news = news_display.style.applymap(color_sentiment, subset=["sentiment_label"])
            st.dataframe(styled_news, use_container_width=True, hide_index=True)
        else:
            st.dataframe(news_display, use_container_width=True, hide_index=True)

    # --- PREDICTION HISTORY SECTION ---
    st.markdown("---")
    st.subheader("📅 Historical Prediction Tracking")
    st.markdown("Track how well the models have been performing over time. The **Actual Close Price** is updated on the following trading day to compare against the predictions made.")
    
    history_path = get_project_root() / "data" / "prediction_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        if not history_df.empty:
            # Sort by date descending
            history_df = history_df.sort_values(by="Prediction_Date", ascending=False).reset_index(drop=True)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("Prediction history is currently empty.")
    else:
        st.info("Prediction history file not found. Predictions will be recorded after the next run.")

# Footer
st.markdown("---")
st.markdown("**AdaniGreenPredictor** | Built with PyTorch Lightning, PyTorch Forecasting & Streamlit")
