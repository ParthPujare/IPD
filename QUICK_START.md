# Quick Start Guide - Running AdaniGreenPredictor

## ✅ Issue Fixed: Import Error

The `ModuleNotFoundError: No module named 'pytorch_lightning'` has been fixed by:
1. Making imports lazy (only load when needed)
2. Ensuring venv Python is used when launching Streamlit

## 🚀 How to Run (Windows)

### Option 1: Using main.py (Recommended)

```powershell
# 1. Activate virtual environment
venv\Scripts\Activate.ps1

# 2. Run main.py (it will fetch data and launch dashboard)
python main.py
```

### Option 2: Direct Streamlit Run

```powershell
# 1. Activate virtual environment
venv\Scripts\Activate.ps1

# 2. Run Streamlit directly
python -m streamlit run app/dashboard.py
```

**Important**: Always use `python -m streamlit` (not just `streamlit`) to ensure venv Python is used.

## 🔧 If You Still See Import Errors

If you still see import errors, ensure you're using the venv Python:

```powershell
# Check which Python is being used
venv\Scripts\python.exe -c "import sys; print(sys.executable)"

# Verify pytorch-lightning is installed in venv
venv\Scripts\python.exe -c "import pytorch_lightning; print('OK')"
```

If the second command fails, reinstall dependencies:

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 📊 Dashboard Features (Without Models)

Even if models aren't trained yet, the dashboard will show:
- ✅ Stock price charts
- ✅ Latest news with sentiment
- ✅ Data metrics
- ⚠️ Predictions (will show error message, but dashboard still works)

## 🎯 Next Steps

1. **View dashboard**: Should open at `http://localhost:8501`
2. **Train models**: Transfer to Mac M4 and train models
3. **Copy models back**: Transfer `models/` directory to Windows
4. **Run predictions**: Dashboard will then show predictions

## ⚠️ Note About yfinance Error

You may see:
```
ADANIGREEN.NS: No data found for this date range, symbol may be delisted
```

This is a yfinance API issue. The dashboard will use existing data from `data/stock_data.csv` which was already fetched successfully (1,237 records).

---

**The dashboard should now run successfully!** 🎉

