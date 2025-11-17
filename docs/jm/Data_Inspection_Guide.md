# Jump Model - Data Inspection and Outlier Check Guide

## Files Created and Ready for Review

### 1. Visualizations Generated ✅
Located in: `docs/jm/figures/`

- **`jumps_simple.png`** - Complete jump detection visualization showing:
  - S&P 500 price with detected jumps marked in red
  - Daily returns with jump highlights
  - VIX levels with HMM threshold comparison
  
### 2. Data Files for Analysis ✅
Located in: `data/jm/`

- **`jump_indicators_simple.csv`** - 333 detected jumps with full metrics
- **`jump_regime_labels.csv`** - Daily regime labels + jump features (9,020 rows)
- **`jump_regime_segments.csv`** - 165 continuous regime segments
- **`tasks_metadata.json`** - 23 MAML tasks generated
- **`task_split_indices_jm.json`** - Train/val/test splits

### 3. Interactive Notebook Created ✅
**`notebooks/jm/JM_Data_Inspection_and_Outlier_Detection.ipynb`**

## How to Inspect Data and Check for Outliers

### Option 1: Open the Jupyter Notebook (Recommended)
```powershell
# Navigate to project root
cd C:\Users\Hp\Desktop\maml-dynamic-portfolio-allocation

# Open the notebook in VS Code
code notebooks/jm/JM_Data_Inspection_and_Outlier_Detection.ipynb
```

The notebook includes sections for:
1. Loading and inspecting jump data
2. Statistical outlier detection (IQR, Z-score)
3. Visualization of jump distributions
4. Missing data analysis
5. Data quality validation

### Option 2: Quick Command-Line Inspection

#### Check Jump Statistics
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/jm/jump_indicators_simple.csv'); print('\n=== JUMP STATISTICS ==='); print(f'Total jumps: {df[\"jump_flag\"].sum()}'); print(f'Jump frequency: {df[\"jump_flag\"].mean():.2%}'); print(f'\nJump size distribution:'); print(df[df[\"jump_flag\"]==1][\"jump_size\"].describe()); print(f'\n=== OUTLIER CHECK (>3 std dev) ==='); jumps = df[df[\"jump_flag\"]==1][\"jump_size\"]; z_scores = abs((jumps - jumps.mean()) / jumps.std()); outliers = jumps[z_scores > 3]; print(f'Potential outliers: {len(outliers)}'); print(outliers.sort_values())"
```

#### Check Regime Distribution
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/jm/jump_regime_labels.csv'); print('\n=== REGIME DISTRIBUTION ==='); print(df['regime_label'].value_counts().sort_index()); print(f'\n=== JUMP INTENSITY STATS ==='); print(df.groupby('regime_label')['jump_intensity'].describe())"
```

#### Check for Missing Data
```powershell
python -c "import pandas as pd; jump_df = pd.read_csv('data/jm/jump_indicators_simple.csv'); regime_df = pd.read_csv('data/jm/jump_regime_labels.csv'); print('\n=== MISSING DATA CHECK ==='); print('\nJump Indicators:'); print(jump_df.isnull().sum()); print('\nRegime Labels:'); print(regime_df.isnull().sum())"
```

### Option 3: Visual Inspection of Generated Images

#### View Jump Detection Visualization
```powershell
# Open the visualization in default image viewer
ii docs/jm/figures/jumps_simple.png
```

This shows:
- Red dots = Detected jumps on S&P 500 chart
- Jump returns highlighted in bar chart
- VIX levels with threshold lines (40 = HMM, 30 = alternative)

## Key Data Quality Checks to Perform

### 1. Check for Extreme Outliers in Jump Sizes

**Expected:**
- Most jumps between 2.5% - 6%
- A few extreme events: 2020 COVID crash (-11.98%), 2008 crisis (-9%)

**Red Flags:**
- Jumps > 15% (data error)
- Jumps on non-trading days
- Impossible return values (>20%)

**Run this check:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/jm/jump_indicators_simple.csv', parse_dates=['Date'], index_col='Date'); jumps = df[df['jump_flag']==1].copy(); print('\n=== EXTREME JUMPS (>10%) ==='); extreme = jumps[abs(jumps['jump_size']) > 0.10]; print(extreme[['jump_size', 'abs_return']].sort_values('jump_size'))"
```

### 2. Check for Jump Clustering Anomalies

**Expected:**
- 2008/2009/2020 should show high clustering (>0.5)
- Normal periods should have low clustering (<0.2)

**Red Flags:**
- Clustering > 1.0 (impossible)
- NaN or inf values

**Run this check:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/jm/jump_regime_labels.csv'); print('\n=== CLUSTERING STATS ==='); print(df['jump_clustering'].describe()); print(f'\nNaN values: {df[\"jump_clustering\"].isna().sum()}'); print(f'Inf values: {np.isinf(df[\"jump_clustering\"]).sum()}'); print(f'\nValues > 1.0: {(df[\"jump_clustering\"] > 1.0).sum()}')"
```

### 3. Check for Date Gaps

**Expected:**
- Weekends/holidays excluded (normal)
- ~252 trading days per year

**Red Flags:**
- Multi-week gaps in trading days
- Duplicate dates

**Run this check:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/jm/jump_regime_labels.csv', parse_dates=['Date'], index_col='Date'); gaps = df.index.to_series().diff(); large_gaps = gaps[gaps > pd.Timedelta(days=7)]; print(f'\n=== DATE GAPS > 7 DAYS ==='); print(f'Total gaps: {len(large_gaps)}'); if len(large_gaps) > 0: print(large_gaps.head(10))"
```

### 4. Validate Ukraine 2022 Detection

**Expected:**
- 5 jumps detected in Feb-Apr 2022
- JR2 regime segment Apr 29 - Jun 3, 2022

**Run this check:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/jm/jump_indicators_simple.csv', parse_dates=['Date'], index_col='Date'); ukraine = df[(df.index >= '2022-02-15') & (df.index <= '2022-04-30')]; print('\n=== UKRAINE 2022 (Feb 15 - Apr 30) ==='); print(f'Total days: {len(ukraine)}'); print(f'Jumps detected: {ukraine[\"jump_flag\"].sum()}'); print(f'Jump dates:'); jumps_ukraine = ukraine[ukraine[\"jump_flag\"]==1]; for date, row in jumps_ukraine.iterrows(): print(f'  {date.date()}: {row[\"jump_size\"]:.2%}')"
```

### 5. Compare with HMM Detection

**Expected:**
- JM detects Ukraine 2022, HMM doesn't
- JM detects 2018 December, HMM doesn't
- Both detect 2008, 2020

**Run this check:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/jm/jump_indicators_simple.csv', parse_dates=['Date'], index_col='Date'); df.columns = df.columns.str.lower(); crises = [('2008 Crisis', '2008-09-01', '2008-12-31'), ('2020 COVID', '2020-02-15', '2020-04-30'), ('2022 Ukraine', '2022-02-15', '2022-04-30'), ('2018 December', '2018-12-01', '2018-12-31')]; print('\n=== HMM VS JM COMPARISON ==='); for name, start, end in crises: period = df[(df.index >= start) & (df.index <= end)]; jm_jumps = period['jump_flag'].sum(); hmm_days = (period['vix_close'] > 40).sum(); print(f'\n{name}:'); print(f'  JM jumps: {jm_jumps}'); print(f'  HMM days (VIX>40): {hmm_days}'); if jm_jumps > 0 and hmm_days == 0: print(f'  ✅ JM DETECTS, HMM MISSES!')"
```

## Common Data Issues and How to Fix

### Issue 1: NaN in Jump Clustering
**Cause:** Autocorrelation calculation fails with <2 observations
**Fix:** Already handled in script (fills with 0)
**Check:** Look for inf or >1.0 values

### Issue 2: Extremely Large Jumps (>15%)
**Cause:** Data errors or flash crashes
**Action:** Investigate specific dates, cross-reference with news
**Example:** 
```powershell
# Check if 2025-04-09 jump (+9.52%) is real or data error
python -c "import pandas as pd; df = pd.read_csv('data/processed/sp500_vix_merged_clean.csv', parse_dates=['Date'], index_col='Date'); print(df.loc['2025-04-08':'2025-04-10'])"
```

### Issue 3: Missing Regime Labels
**Cause:** Early data before rolling window fills
**Expected:** First ~20 days may have incomplete metrics
**Fix:** Already filtered in task generation (start from 1990-03-13)

### Issue 4: Low Task Count (23 vs Expected 650)
**Cause:** Data has weekends/holidays, only ~28-29 trading days per 40-calendar-day window
**Fix:** Relaxed threshold to 75% (30 trading days minimum)
**Result:** Now creates 23 valid tasks

## Visualization Checklist

Open `docs/jm/figures/jumps_simple.png` and verify:

### Panel 1: S&P 500 with Jumps
- ✅ Red dots appear at major crashes (2008, 2020)
- ✅ Some positive jumps (rallies) also marked
- ✅ Ukraine 2022 should show ~5 red dots

### Panel 2: Returns with Jumps
- ✅ Large bars (>2.5%) are highlighted in red
- ✅ Largest negative: 2020-03-16 (-11.98%)
- ✅ Largest positive: 2008-10-13 (+11.58%)

### Panel 3: VIX with Thresholds
- ✅ Red line at VIX 40 (HMM threshold)
- ✅ Orange line at VIX 30 (alternative)
- ✅ Ukraine 2022 shows VIX ~30-36 (below HMM, above alternative)

## Data Quality Summary (Based on Script Output)

From the jump detection run:

✅ **PASS: Jump Frequency**
- 333 jumps / 9,020 days = 3.69% (reasonable)
- Avg 27 days between jumps (expected ~20-30)

✅ **PASS: Jump Direction**
- 54.4% negative, 45.6% positive (reasonable asymmetry)
- Crashes slightly more common than rallies

✅ **PASS: Crisis Detection**
- 2008: 44 jumps (51.8% frequency) ✅
- 2020: 28 jumps (53.8% frequency) ✅
- 2022: 5 jumps (9.6% frequency) ✅
- 2018: 3 jumps ✅

✅ **PASS: Extreme Events**
- Largest crash: -11.98% (2020-03-16 COVID) - Real event ✅
- Largest rally: +11.58% (2008-10-13 Crisis bounce) - Real event ✅

⚠️ **CHECK NEEDED: 2025-04-09 Jump (+9.52%)**
- This is future data (2025)
- May be synthetic or test data
- Verify if this is real or should be excluded

## Recommended Actions

1. **Open the notebook** and run all cells to see full analysis
2. **View the visualization** (`ii docs/jm/figures/jumps_simple.png`)
3. **Run the outlier checks** above to verify data quality
4. **Investigate 2025 data** - may need to truncate to present day
5. **Review task generation** - 23 tasks seems low, may need different stride

## Next Steps After Validation

Once data quality is confirmed:
1. ✅ Fix any identified outliers or errors
2. ✅ Re-run task generation with optimized parameters
3. ✅ Create MAML training script for JM tasks
4. ✅ Train JM-MAML and compare with HMM-MAML
5. ✅ Test on Ukraine 2022 holdout

---

**Bottom Line:** The Jump Model data looks good overall. Main checks needed:
1. Verify 2025 data is intentional (or truncate to 2024)
2. Confirm no extreme outliers (>15% jumps are real events)
3. Validate Ukraine 2022 detection (5 jumps found ✅)
