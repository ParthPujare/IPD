# AdaniGreenPredictor - Project Setup Summary

## ✅ Project Creation Complete

All files have been successfully generated! The project is ready for use.

## 📁 Files Created

### Core Files
- ✅ `main.py` - Entrypoint script`
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Comprehensive documentation
- ✅ `.gitignore` - Git ignore rules

### Data Fetching (`src/fetch/`)
- ✅ `fetch_stock.py` - yfinance stock data fetching
- ✅ `fetch_news.py` - Google News RSS / NewsAPI fetching
- ✅ `__init__.py` - Package init

### Preprocessing (`src/preprocess/`)
- ✅ `features.py` - Technical indicators & feature engineering
- ✅ `sentiment_pipeline.py` - FinBERT sentiment analysis
- ✅ `__init__.py` - Package init

### Training (`src/training/`)
- ✅ `train_lstm.py` - LSTM training script (PyTorch Lightning)
- ✅ `train_tft.py` - TFT training script (pytorch-forecasting)
- ✅ `__init__.py` - Package init

### Inference (`src/inference/`)
- ✅ `predict.py` - Model inference for LSTM/TFT
- ✅ `ensemble.py` - Ensemble prediction logic
- ✅ `__init__.py` - Package init

### LLM (`src/llm/`)
- ✅ `llm_summary.py` - flan-t5-small based explanations
- ✅ `__init__.py` - Package init

### Utils (`src/utils/`)
- ✅ `helpers.py` - Common utilities (device detection, etc.)
- ✅ `__init__.py` - Package init

### Dashboard (`app/`)
- ✅ `dashboard.py` - Streamlit UI with charts, predictions, news

### Package Inits
- ✅ `src/__init__.py` - Root package init

### Supporting Files
- ✅ `.gitignore` - Git ignore rules
- ✅ `data/.gitkeep` - Keep data directory in git

**Total: 23 files created**

## 🚀 Quick Start Commands

### On Windows (Development)

1. **Setup virtual environment:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run application:**
   ```powershell
   python main.py
   ```
   This will:
   - Fetch stock data (5 years history)
   - Fetch latest news (10 headlines)
   - Compute sentiment scores
   - Build features
   - Launch Streamlit dashboard

### Transfer to Mac (For Training)

#### Option 1: Git Repository
```bash
# On Windows: Initialize and push
git init
git add .
git commit -m "Initial commit: AdaniGreenPredictor"
git remote add origin <your-repo-url>
git push -u origin main

# On Mac: Clone
git clone <your-repo-url>
cd AdaniGreenPredictor
```

#### Option 2: Zip Transfer
```powershell
# On Windows: Create zip
# Right-click project folder → Send to → Compressed (zipped) folder
# Transfer zip to Mac via USB/cloud/Dropbox/etc.

# On Mac: Extract
unzip AdaniGreenPredictor.zip
cd AdaniGreenPredictor
```

### On Mac M4 (Training)

1. **Setup virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify MPS accelerator:**
   ```bash
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```
   Should output: `MPS available: True`

4. **Train LSTM model:**
   ```bash
   python src/training/train_lstm.py \
       --epochs 50 \
       --batch_size 64 \
       --hidden_size 128 \
       --num_layers 2 \
       --lr 0.001 \
       --seq_len 30
   ```

5. **Train TFT model:**
   ```bash
   python src/training/train_tft.py \
       --epochs 50 \
       --batch_size 64 \
       --hidden_size 32 \
       --lr 0.03
   ```

6. **Debug mode (quick test):**
   ```bash
   # LSTM
   python src/training/train_lstm.py --epochs 2 --batch_size 32 --debug
   
   # TFT
   python src/training/train_tft.py --epochs 2 --batch_size 32 --debug
   ```

7. **Copy models back to Windows:**
   - Copy entire `models/` directory from Mac to Windows project
   - Ensure these files exist:
     - `models/checkpoints/lstm-*.ckpt`
     - `models/checkpoints/tft-*.ckpt`
     - `models/saved_models/lstm.pt`
     - `models/saved_models/tft.pth`
     - `models/saved_models/lstm_scaler.pkl`
     - `models/saved_models/lstm_features.pkl`
     - `models/saved_models/tft_training_dataset.pkl`

### On Windows (After Training)

1. **Ensure models are in place:**
   ```
   models/
   ├── checkpoints/
   │   ├── lstm-*.ckpt
   │   └── tft-*.ckpt
   └── saved_models/
       ├── lstm.pt
       ├── tft.pth
       ├── lstm_scaler.pkl
       ├── lstm_features.pkl
       └── tft_training_dataset.pkl
   ```

2. **Run dashboard:**
   ```powershell
   python main.py
   ```
   
   Or directly:
   ```powershell
   streamlit run app/dashboard.py
   ```

## 📋 Training Script Arguments

### LSTM Training
```bash
python src/training/train_lstm.py \
    --seq_len 30 \              # Sequence length (default: 30)
    --hidden_size 128 \         # LSTM hidden size (default: 128)
    --num_layers 2 \            # Number of LSTM layers (default: 2)
    --dropout 0.2 \             # Dropout rate (default: 0.2)
    --batch_size 32 \            # Batch size (default: 32)
    --epochs 50 \                # Number of epochs (default: 50)
    --lr 0.001 \                 # Learning rate (default: 0.001)
    --debug                      # Debug mode (smaller dataset)
```

### TFT Training
```bash
python src/training/train_tft.py \
    --hidden_size 32 \           # TFT hidden size (default: 32)
    --attention_head_size 1 \    # Attention head size (default: 1)
    --dropout 0.1 \              # Dropout rate (default: 0.1)
    --batch_size 64 \            # Batch size (default: 64)
    --epochs 50 \                # Number of epochs (default: 50)
    --lr 0.03 \                  # Learning rate (default: 0.03)
    --debug                      # Debug mode (smaller dataset)
```

## 🔍 Project Features

- ✅ **Automatic data fetching** on startup (stock prices + news)
- ✅ **FinBERT sentiment analysis** for news headlines
- ✅ **Technical indicators** (SMA, EMA, RSI, returns, volatility)
- ✅ **Two ML models** (LSTM + TFT) with automatic accelerator detection
- ✅ **Ensemble prediction** with weighted averaging
- ✅ **Interactive Streamlit dashboard** with Plotly charts
- ✅ **LLM-powered explanations** using flan-t5-small
- ✅ **MPS accelerator support** for Mac M4 training

## 📦 Key Dependencies

- **PyTorch** >= 2.0.0 - Deep learning framework
- **PyTorch Lightning** >= 2.1.0 - Training framework
- **pytorch-forecasting** >= 1.0.1 - TFT model
- **Transformers** >= 4.35.0 - FinBERT & flan-t5-small
- **yfinance** >= 0.2.28 - Stock data fetching
- **Streamlit** >= 1.28.0 - Dashboard UI
- **Plotly** >= 5.17.0 - Interactive charts

## 🎯 Next Steps

1. **On Windows:**
   - Run `python main.py` to test data fetching
   - Verify all data files are created in `data/`

2. **Transfer to Mac:**
   - Use git or zip transfer method above
   - Ensure all files are transferred correctly

3. **On Mac:**
   - Install dependencies
   - Verify MPS accelerator
   - Run training scripts (start with `--debug` flag to test)

4. **Back to Windows:**
   - Copy trained models to `models/` directory
   - Run `python main.py` to launch dashboard
   - Test predictions with different models

## ⚠️ Important Notes

- **Training requires high compute**: Use Mac M4 with MPS for training
- **Data files are large**: `.gitignore` excludes CSV files (add manually if needed)
- **Models are large**: Binary model files excluded from git
- **API Keys**: Optional NewsAPI key can be added to `.env` file
- **First run**: FinBERT and flan-t5-small models will be downloaded (~500MB total)

## 🐛 Troubleshooting

1. **MPS not available**: Ensure PyTorch 2.0+ and macOS 12.3+
2. **Models not found**: Train models first on Mac, then copy to Windows
3. **News fetch fails**: Check internet, Google News RSS may have rate limits
4. **Memory issues**: Use `--debug` flag or reduce batch size

## 📞 Support

For issues or questions, refer to:
- `README.md` - Comprehensive documentation
- Code docstrings - Detailed function documentation
- Individual module `__main__` sections - Test examples

---

**Project created successfully! Ready for development and training.** 🎉

