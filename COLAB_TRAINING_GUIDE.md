# Google Colab Training Guide

## 🚀 Train Models on Google Colab

This guide shows you how to train LSTM and TFT models on Google Colab for free GPU access.

## 📋 Prerequisites

1. Google account
2. Access to Google Colab
3. Project files uploaded to Google Drive

## 📁 Step 1: Prepare Files

### Option A: Upload via Google Drive

1. **Zip your project** (excluding `venv` and large files):
   ```powershell
   # On Windows, create a zip of your project
   # Exclude: venv/, __pycache__/, *.pyc, models/checkpoints/*.ckpt
   ```

2. **Upload to Google Drive**:
   - Go to [Google Drive](https://drive.google.com)
   - Create folder: `AdaniGreenPredictor`
   - Upload the zipped project
   - Extract in Drive

### Option B: Use Git

```bash
# On Windows
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main

# In Colab, clone the repo
!git clone <your-repo-url>
```

## 🔧 Step 2: Open Colab Notebook

1. **Open the notebook**:
   - Go to [Google Colab](https://colab.research.google.com)
   - Click "File" → "Upload notebook"
   - Upload `notebooks/train_on_colab.ipynb`

2. **Or create new notebook**:
   - Copy code from `notebooks/train_on_colab.ipynb`
   - Paste into new Colab notebook

## 🚀 Step 3: Run Training

### Setup (Run Once)

```python
# Cell 1: Install dependencies
!pip install -q pandas numpy yfinance feedparser requests plotly scikit-learn torch pytorch-lightning pytorch-forecasting transformers sentencepiece python-dotenv accelerate

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Set project path
import os
PROJECT_PATH = '/content/drive/MyDrive/AdaniGreenPredictor'  # Update this!
os.chdir(PROJECT_PATH)
print(f"Current directory: {os.getcwd()}")

# Cell 4: Verify data
import pandas as pd
df = pd.read_csv("data/features.csv")
print(f"✓ Data: {len(df)} rows, {len(df.columns)} columns")
```

### Enable GPU (Important!)

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4 or better)
3. Click **Save**

### Train Models

```python
# Cell 5: Train LSTM (20-30 minutes)
!python src/training/train_lstm.py \
    --epochs 50 \
    --batch_size 64 \
    --hidden_size 128 \
    --num_layers 2 \
    --lr 0.001 \
    --seq_len 30

# Cell 6: Train TFT (20-30 minutes)
!python src/training/train_tft.py \
    --epochs 50 \
    --batch_size 64 \
    --hidden_size 32 \
    --lr 0.03
```

## 💾 Step 4: Download Models

After training completes:

```python
# Cell 7: Zip models folder
!zip -r models.zip models/

# Download from Colab Files sidebar
```

**Or download directly from Drive:**
- Models are saved in Google Drive
- Navigate to `AdaniGreenPredictor/models/`
- Download entire `models/` folder

## 📦 Step 5: Transfer to Local Machine

1. **Download `models.zip`** from Colab Files sidebar
2. **Extract to your local project**:
   ```powershell
   # On Windows
   # Extract models.zip to your project root
   # Should create: models/checkpoints/ and models/saved_models/
   ```
3. **Verify models**:
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

## ✅ Step 6: Test on Local Machine

```powershell
# On Windows
venv\Scripts\Activate.ps1
python main.py
```

The dashboard should now show predictions!

## 📊 Colab Tips

### Monitor Training

- **Watch progress**: Check cell outputs for training metrics
- **Early stopping**: Models stop automatically if validation loss doesn't improve
- **Save checkpoints**: Models auto-save best checkpoints

### Troubleshooting

**Issue: Out of memory**
- Solution: Reduce batch size (`--batch_size 32` instead of 64)

**Issue: Training too slow**
- Solution: Enable GPU (Runtime → Change runtime type → GPU)

**Issue: Can't find data files**
- Solution: Check `PROJECT_PATH` matches your Drive folder location

**Issue: Models not saving**
- Solution: Ensure write permissions in Google Drive folder

## 🎯 Expected Output

After training, you should see:

```
✓ LSTM checkpoint: models/checkpoints/lstm-epoch=XX-val_loss=X.XXXX.ckpt
✓ TFT checkpoint: models/checkpoints/tft-epoch=XX-val_loss=X.XXXX.ckpt
✓ Saved models in models/saved_models/
```

## 💡 Notes

- **Training time**: ~20-30 minutes per model on GPU
- **Model size**: ~10-50 MB per model
- **GPU usage**: Free tier includes limited GPU hours
- **Data persistence**: Models saved in Drive persist between sessions

---

**Now you can train models on Colab and use them locally!** 🎉

