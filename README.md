# AdaniGreenPredictor 📈

A comprehensive machine learning project for predicting Adani Green Energy (ADANIGREEN.NS) stock prices using LSTM and Temporal Fusion Transformer (TFT) models, enhanced with sentiment analysis and LLM-powered explanations.

##  Project Overview

This project implements a full-stack ML pipeline for stock price prediction with:

- **Data Fetching**: Automatic retrieval of historical stock prices and latest news
- **Sentiment Analysis**: FinBERT-based sentiment scoring for news headlines
- **Feature Engineering**: Technical indicators (SMA, EMA, RSI, returns, volatility) + sentiment features
- **Two ML Models**:
  1. **LSTM** (PyTorch Lightning) - Sequence-based prediction
  2. **TFT** (Temporal Fusion Transformer) - Attention-based forecasting with covariates
- **Ensemble Prediction**: Weighted averaging of multiple models
- **Streamlit Dashboard**: Interactive UI with charts, predictions, news, and AI explanations
- **LLM Summaries**: flan-t5-small based explanations of predictions

##  Project Structure

```
AdaniGreenPredictor/
│
├── data/
│   ├── stock_data.csv          # Historical OHLCV data
│   ├── news_data.csv            # News headlines with sentiment
│   └── features.csv             # Engineered features for training
│
├── models/
│   ├── checkpoints/             # Training checkpoints (LSTM, TFT)
│   └── saved_models/            # Final models for inference
│
├── notebooks/                   # Optional EDA notebooks
│
├── src/
│   ├── fetch/
│   │   ├── fetch_stock.py       # yfinance data fetching
│   │   └── fetch_news.py        # Google News RSS / NewsAPI fetching
│   │
│   ├── preprocess/
│   │   ├── features.py           # Technical indicators & feature engineering
│   │   └── sentiment_pipeline.py # FinBERT sentiment analysis
│   │
│   ├── training/
│   │   ├── train_lstm.py        # LSTM training script
│   │   └── train_tft.py         # TFT training script
│   │
│   ├── inference/
│   │   ├── predict.py           # Model inference (LSTM/TFT)
│   │   └── ensemble.py          # Ensemble prediction logic
│   │
│   ├── llm/
│   │   └── llm_summary.py       # flan-t5-small explanations
│   │
│   └── utils/
│       └── helpers.py           # Common utilities (device detection, etc.)
│
├── app/
│   └── dashboard.py             # Streamlit dashboard UI
│
├── main.py                      # Entry point (refreshes data + launches dashboard)
├── requirements.txt             # Python dependencies
└── README.md                     # This file
```

##  Setup Instructions

### Prerequisites

- Python 3.8+
- Windows (development environment)
- Mac M4 (for training - supports MPS accelerator)

### Installation on Windows (Development)

1. **Clone or download the project**:
   ```bash
   cd "E:\1.Parth\Personal\2.Education\4. Engeneering\1. DJ Sanghvi\3rd Year\IPD\Final Implementation"
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application** (fetches data and launches dashboard):
   ```bash
   python main.py
   ```

   The dashboard will automatically open in your browser at `http://localhost:8501`

### Transfer to Mac for Training

#### Option 1: Git Repository
```bash
# On Windows: Commit and push to git
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main

# On Mac: Clone the repository
git clone <your-repo-url>
cd AdaniGreenPredictor
```

#### Option 2: Zip Transfer
```bash
# On Windows: Create zip file
# Right-click project folder → Send to → Compressed (zipped) folder
# Transfer zip file to Mac via USB/cloud/etc.

# On Mac: Extract and navigate
unzip AdaniGreenPredictor.zip
cd AdaniGreenPredictor
```

### Installation on Mac (Training Environment)

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify MPS (Metal Performance Shaders) availability**:
   ```python
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

   Should output: `MPS available: True`

## 🏋️ Training Models

### Training on Mac M4

**Note**: Training is computationally intensive. Use a Mac M4 with MPS acceleration for best performance.

#### Train LSTM Model

```bash
python src/training/train_lstm.py \
    --epochs 50 \
    --batch_size 64 \
    --hidden_size 128 \
    --num_layers 2 \
    --lr 0.001 \
    --seq_len 30
```

**Debug mode** (quick test with small dataset):
```bash
python src/training/train_lstm.py --epochs 2 --batch_size 32 --debug
```

#### Train TFT Model

```bash
python src/training/train_tft.py \
    --epochs 50 \
    --batch_size 64 \
    --hidden_size 32 \
    --lr 0.03
```

**Debug mode**:
```bash
python src/training/train_tft.py --epochs 2 --batch_size 32 --debug
```

### Training Output

Models will be saved to:
- **Checkpoints**: `models/checkpoints/lstm-*.ckpt`, `models/checkpoints/tft-*.ckpt`
- **Final models**: `models/saved_models/lstm.pt`, `models/saved_models/tft.pth`

After training, copy the `models/` directory back to Windows for inference.

##  Running the Dashboard

### On Windows (After Training)

1. **Ensure trained models are in `models/saved_models/`**:
   - `lstm.pt` (LSTM model)
   - `tft.pth` (TFT model)
   - `lstm_scaler.pkl` (LSTM scaler)
   - `lstm_features.pkl` (LSTM feature list)
   - `tft_training_dataset.pkl` (TFT dataset)

2. **Run the dashboard**:
   ```bash
   python main.py
   ```

   Or directly with Streamlit:
   ```bash
   streamlit run app/dashboard.py
   ```

3. **Features**:
   - Select model: LSTM, TFT, or Ensemble
   - View interactive price charts with predictions
   - See latest news with sentiment analysis
   - Read AI-generated explanations
   - Refresh data on demand

##  Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root for API keys:

```env
# Optional: NewsAPI key for better news fetching
NEWSAPI_KEY=your_newsapi_key_here
```

If not provided, the project will use Google News RSS feeds (free, no API key required).

##  Usage Examples

### Fetch Data Only

```python
from src.fetch.fetch_stock import update_stock_data
from src.fetch.fetch_news import update_news_data

# Fetch stock data
update_stock_data()

# Fetch news data
update_news_data()
```

### Compute Sentiment

```python
from src.preprocess.sentiment_pipeline import add_sentiment_to_news_data

# Compute sentiment for news headlines
add_sentiment_to_news_data()
```

### Build Features

```python
from src.preprocess.features import build_features

# Build features with technical indicators + sentiment
features_df = build_features()
```

### Make Predictions

```python
from src.inference.predict import predict_next_day
from src.inference.ensemble import ensemble_predict

# LSTM prediction
lstm_result = predict_next_day("LSTM")
print(f"Predicted price: ₹{lstm_result['predicted_price']:.2f}")

# TFT prediction
tft_result = predict_next_day("TFT")
print(f"Predicted price: ₹{tft_result['predicted_price']:.2f}")

# Ensemble prediction
ensemble_result = ensemble_predict(weights={"LSTM": 0.5, "TFT": 0.5})
print(f"Ensemble predicted price: ₹{ensemble_result['predicted_price']:.2f}")
```

##  Troubleshooting

### Issue: Models not found

**Solution**: Train models first on Mac, then copy `models/` directory to Windows.

### Issue: MPS not available on Mac

**Solution**: Ensure you're using PyTorch 2.0+ and macOS 12.3+. Verify with:
```python
import torch
print(torch.backends.mps.is_available())
```

### Issue: News fetching fails

**Solution**: Check internet connection. Google News RSS may have rate limits. Consider using NewsAPI with a valid key.

### Issue: FinBERT model download fails

**Solution**: Ensure stable internet connection. The model will be downloaded automatically on first run (~440MB).

##  Dependencies

Key packages:
- `torch` >= 2.0.0 - PyTorch
- `pytorch-lightning` >= 2.1.0 - Training framework
- `pytorch-forecasting` >= 1.0.1 - TFT model
- `transformers` >= 4.35.0 - FinBERT & flan-t5-small
- `yfinance` >= 0.2.28 - Stock data fetching
- `streamlit` >= 1.28.0 - Dashboard UI
- `plotly` >= 5.17.0 - Interactive charts

See `requirements.txt` for complete list.

##  License

This project is for educational purposes. Stock predictions are for demonstration only and should not be used for actual trading decisions.

##  Contributing

This is a project implementation. Feel free to extend and improve:

- Add more technical indicators
- Experiment with different architectures
- Improve sentiment analysis
- Add more data sources
- Enhance dashboard features

##  Contact

For questions or issues, please check the code documentation or create an issue in the repository.

---


