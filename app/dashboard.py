import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[1] 
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_project_root
from src.inference.predict import predict_next_day
from src.llm.llm_summary import generate_summary 

NEWS_DATA_PATH = get_project_root() / "data" / "news_data.csv"
STOCK_DATA_PATH = get_project_root() / "data" / "stock_data.csv"

# --- UI STYLING ---
def apply_sentiment_style(val):
    if not val: return ""
    label = str(val).strip().lower()
    if "positive" in label:
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    elif "negative" in label:
        return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
    return "background-color: #fff3cd; color: #856404; font-weight: bold;"

# --- DATA LOADERS ---
def load_news_data(ticker):
    if NEWS_DATA_PATH.exists():
        df = pd.read_csv(NEWS_DATA_PATH)
        df = df.rename(columns={'sentiment_sentiment_score': 'sentiment_score', 'sentiment_score.1': 'sentiment_score'})
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0.0)
        return df[df["ticker"] == ticker].copy()
    return pd.DataFrame()

def load_stock_data(ticker):
    if STOCK_DATA_PATH.exists():
        df = pd.read_csv(STOCK_DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df[df["ticker"] == ticker].copy()
    return pd.DataFrame()

# --- DASHBOARD RENDER ---
st.set_page_config(page_title="StockPredictor Pro", layout="wide")
st.title("Multi-Stock Predictor Dashboard")

# Sidebar
st.sidebar.title("Settings")
selected_ticker = st.sidebar.selectbox("Select Stock", ["ADANIGREEN.NS", "BEL.NS", "HDFCBANK.NS", "INDIGO.NS", "VEDL.NS"])
model_option = st.sidebar.selectbox("Select Model", ["LSTM", "TFT", "Ensemble"])

if st.sidebar.button("Refresh Data (Manual)"):
    st.rerun()

stock_df = load_stock_data(selected_ticker)
news_df = load_news_data(selected_ticker)

if not stock_df.empty:
    # 1. Top Metrics (4-Column Layout)
    last_price = stock_df['Close'].iloc[-1]
    prev_price = stock_df['Close'].iloc[-2]
    price_change = last_price - prev_price
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Last Price", f"INR {last_price:.2f}")
    with col2:
        st.metric("1-Day Change", f"INR {price_change:.2f}", delta=f"{price_change:.2f}")
    with col3:
        st.metric("Ticker", selected_ticker)
    with col4:
        current_sent = "Neutral"
        if not news_df.empty:
            score = news_df['sentiment_score'].iloc[0]
            if score > 0.05: current_sent = "Positive"
            elif score < -0.05: current_sent = "Negative"
        st.metric("Sentiment", current_sent)

    # 2. Forecast Section
    st.markdown("---")
    st.subheader(f"Next-Day Prediction ({model_option})")
    res = predict_next_day(model_name=model_option, ticker=selected_ticker)
    
    if res:
        p_price, p_change = res["predicted_price"], res["predicted_change_pct"]
        
        # --- NEW LOGIC: Calculate next trading day ---
        from datetime import datetime, timedelta
        
        # Get the latest date from your data
        last_date = stock_df["date"].max()
        
        # Logic to skip weekends: if Friday, next day is Monday
        if last_date.weekday() == 4: # Friday
            next_trading_day = last_date + timedelta(days=3)
        elif last_date.weekday() == 5: # Saturday
            next_trading_day = last_date + timedelta(days=2)
        else:
            next_trading_day = last_date + timedelta(days=1)
            
        target_date_str = next_trading_day.strftime("%A, %B %d, %Y")
        # ----------------------------------------------

        color = "#f8d7da" if p_change < 0 else "#d4edda"
        text_color = "#721c24" if p_change < 0 else "#155724"
        
        st.markdown(f'''
            <div style="background-color:{color}; padding:20px; border-radius:10px; border: 2px solid {text_color};">
                <h2 style="color:{text_color}; margin:0;">Predicted Price: INR {p_price:.2f} ({p_change:+.2f}%)</h2>
                <p style="color:{text_color}; margin:0; opacity:0.8;">Target Date: {target_date_str}</p>
            </div>
        ''', unsafe_allow_html=True)

    # 3. Timeline Graph with Enhanced Volume Visibility
    st.markdown("---")
    st.subheader(f"{selected_ticker} Price Timeline (Historical to 2026)")
    
    # Create subplots with a shared X-axis and secondary Y-axis for Volume
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price Line (Primary Y-Axis)
    fig.add_trace(go.Scatter(
        x=stock_df["date"], y=stock_df["Close"], 
        name="Close Price", line=dict(color="#007bff", width=2)
    ), secondary_y=False)
    
    # Prediction Point (Star)
    if p_price:
        next_date = stock_df["date"].max() + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[next_date], y=[p_price],
            name="Prediction", mode="markers",
            marker=dict(color="red", size=15, symbol="star")
        ), secondary_y=False)

    # Volume Bars (Secondary Y-Axis)
    fig.add_trace(go.Bar(
        x=stock_df["date"], y=stock_df["Volume"], 
        name="Volume", marker_color="rgba(200, 200, 200, 0.3)", 
    ), secondary_y=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        yaxis2_title="Volume",
        yaxis2_showgrid=False, 
        # Forces volume bars to stay at the bottom 25% of the chart
        yaxis2_range=[0, stock_df["Volume"].max() * 4], 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Sentiment Indicators Table
    st.markdown("---")
    st.subheader("Market Sentiment Intelligence")
    if not news_df.empty:
        def get_label(s):
            if s > 0.05: return "Positive"
            if s < -0.05: return "Negative"
            return "Neutral"
        
        news_df['sentiment_indicator'] = news_df['sentiment_score'].apply(get_label)
        display_df = news_df[["published_date", "title", "sentiment_indicator", "source"]].head(10)
        
        st.dataframe(
            display_df.style.map(apply_sentiment_style, subset=["sentiment_indicator"]),
            use_container_width=True, 
            hide_index=True
        )

    # 5. AI Summary Section
    st.markdown("---")
    st.subheader("Market Summary Analysis")
    
    summary_context = {
        "ticker_name": selected_ticker,
        "predicted_price": p_price if p_price else last_price,
        "last_price": last_price,
        "predicted_change_pct": p_change if res else 0.0,
        "sentiment_summary": current_sent
    }
    
    with st.spinner("Analyzing data..."):
        st.write(generate_summary(summary_context))

else:
    st.error("Missing data files. Please run scrapers first.")