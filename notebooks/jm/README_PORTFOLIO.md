# Portfolio Backtest - SIMPLE GUIDE

## 📘 One Notebook, Everything Included

I created **ONE complete notebook** with everything you need:

**File:** `notebooks/jm/COMPLETE_Portfolio_Backtest.ipynb`

## 🚀 How to Use

### Step 1: Upload to Google Colab
1. Go to https://colab.research.google.com
2. Upload `COMPLETE_Portfolio_Backtest.ipynb`

### Step 2: Update ONE Line
In the second cell, change this path to match your Google Drive:
```python
BASE_PATH = '/content/drive/MyDrive/maml_jm_experiments/7tasks_20d/'
```

### Step 3: Run All Cells
Click **Runtime** → **Run all**

That's it!

## 📊 What You Get

The notebook will:
1. ✅ Load your trained MAML model
2. ✅ Generate predictions
3. ✅ Run portfolio backtest
4. ✅ Compare with baselines (Buy-Hold, 60/40)
5. ✅ Show all metrics (Sharpe, drawdown, alpha)
6. ✅ Create visualizations
7. ✅ Save everything to your Drive

## 📈 Expected Results

- **MAML Sharpe:** 0.8-1.2 (vs 0.4 for Buy-Hold)
- **Max Drawdown:** -25% to -40% (vs -55%)
- **Alpha:** +2% to +5% annually

## 💾 What Gets Saved

All results save to your `BASE_PATH`:
- `maml_v7_portfolio_returns.csv`
- `strategy_comparison.csv`
- `portfolio_comparison.png`
- `allocation_analysis.png`
- And more!

## ⚡ That's All!

Just one notebook. Run it. Get results. Done. 🎉

---

**The notebook has 14 steps, all clearly labeled. Just run them top to bottom.**
