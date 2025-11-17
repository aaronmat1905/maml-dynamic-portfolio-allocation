# Jump Model Pipeline: From Raw Data to MAML Tasks

## 📊 Complete Data Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STEP 1: RAW DATA INPUT                              │
└─────────────────────────────────────────────────────────────────────────────┘

    📁 data/raw/sp500_vix_merged_full.csv
    
    ┌──────────┬──────────┬──────────┬──────────┐
    │   Date   │ S&P 500  │   VIX    │  Return  │
    ├──────────┼──────────┼──────────┼──────────┤
    │1990-01-02│  359.69  │   17.24  │    -     │
    │1990-01-03│  358.76  │   18.19  │ -0.26%   │
    │1990-01-04│  355.67  │   19.22  │ -0.86%   │
    │   ...    │   ...    │   ...    │   ...    │
    │2025-10-23│ 5808.12  │   16.32  │ +0.12%   │
    └──────────┴──────────┴──────────┴──────────┘
    
    Total: 9,020 trading days (1990-2025)


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                   STEP 2: JUMP DETECTION (3.5% Threshold)                    │
│                     Script: scripts/jm/detect_jumps.py                       │
└─────────────────────────────────────────────────────────────────────────────┘

    Method: Simple Threshold Test
    Condition: |return| > 3.5%
    
    ┌──────────┬──────────┬────────────┐
    │   Date   │  Return  │ Jump Flag  │
    ├──────────┼──────────┼────────────┤
    │1990-01-02│ +0.12%   │     0      │  ← Normal day
    │1990-01-03│ -0.26%   │     0      │  ← Normal day
    │1997-10-27│ -6.87%   │     1      │  ← JUMP! (Asian Crisis)
    │2008-09-29│ -8.81%   │     1      │  ← JUMP! (Lehman)
    │2020-03-16│-11.98%   │     1      │  ← JUMP! (COVID)
    │2025-10-23│ +0.12%   │     0      │  ← Normal day
    └──────────┴──────────┴────────────┘
    
    Output: data/jm/jump_indicators_simple.csv
    Result: 126 jumps detected (1.40% of days)


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│              STEP 3: JUMP INTENSITY CALCULATION (Rolling 20-day)             │
│                 Script: scripts/jm/classify_jump_regimes.py                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Formula: Jump Intensity = (# jumps in last 20 days) / 20
    
    Example Timeline:
    
    Day 1-10:  No jumps        → Intensity = 0.00  (0/20)
    Day 11:    1 jump occurs   → Intensity = 0.05  (1/20)
    Day 12-30: No new jumps    → Intensity = 0.05  (1/20)
    Day 31:    Window shifts   → Intensity = 0.00  (jump falls out)
    
    2008 Financial Crisis Example:
    ┌──────────┬───────┬──────────┬───────────────────┐
    │   Date   │ Jump  │  Count   │ Intensity (20d)   │
    ├──────────┼───────┼──────────┼───────────────────┤
    │2008-09-15│   1   │    1     │     0.05          │
    │2008-09-17│   1   │    2     │     0.10          │
    │2008-09-29│   1   │    3     │     0.15          │ ← JR2 threshold!
    │2008-10-07│   1   │    4     │     0.20          │
    │2008-10-15│   1   │    5     │     0.25          │
    │2008-10-28│   1   │    6     │     0.30          │ ← Peak intensity
    │   ...    │  ...  │   ...    │      ...          │
    └──────────┴───────┴──────────┴───────────────────┘


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: REGIME CLASSIFICATION                             │
│                 Script: scripts/jm/classify_jump_regimes.py                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Thresholds:
    • JR0 (Calm):     Intensity < 0.05  (< 1 jump/month)
    • JR1 (Moderate): 0.05 ≤ Intensity < 0.15  (1-3 jumps/month)
    • JR2 (Extreme):  Intensity ≥ 0.15  (> 3 jumps/month)
    
    ┌──────────┬───────────┬──────────────┐
    │   Date   │ Intensity │ Regime Label │
    ├──────────┼───────────┼──────────────┤
    │1990-01-02│   0.00    │      0       │  JR0
    │1997-10-27│   0.05    │      1       │  JR1
    │1997-10-28│   0.10    │      1       │  JR1
    │1998-08-31│   0.15    │      2       │  JR2 ⚠️
    │1998-09-30│   0.20    │      2       │  JR2 ⚠️
    │2008-09-29│   0.30    │      2       │  JR2 ⚠️
    │2020-03-16│   0.40    │      2       │  JR2 ⚠️
    └──────────┴───────────┴──────────────┘
    
    Output: data/jm/jump_regime_labels.csv (9,020 rows)
    
    Distribution:
    • JR0: 7,962 days (88.3%)
    • JR1:   808 days ( 9.0%)
    • JR2:   249 days ( 2.8%)


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                STEP 5: SEGMENT IDENTIFICATION (Continuous Periods)           │
│                 Script: scripts/jm/classify_jump_regimes.py                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Group consecutive days with same regime into segments
    
    Example:
    ┌──────────┬────────┐         ┌─────────────────────────────────┐
    │   Date   │ Regime │         │         Segment Output          │
    ├──────────┼────────┤   →     ├────────┬────────┬────────┬──────┤
    │1990-01-02│   0    │         │ Regime │ Start  │  End   │Length│
    │1990-01-03│   0    │         ├────────┼────────┼────────┼──────┤
    │   ...    │   0    │         │   0    │1990-01 │1991-01 │ 380  │
    │1991-01-17│   0    │         │   1    │1991-01 │1991-02 │  28  │
    │1991-01-18│   1    │ ───→    │   0    │1991-02 │1997-10 │2,142 │
    │1991-02-14│   1    │         │   1    │1997-10 │1997-11 │  29  │
    │1991-02-15│   0    │         │   2    │1998-08 │1998-09 │  30  │ ← JR2!
    │   ...    │   0    │         │   2    │2008-09 │2009-01 │ 110  │ ← JR2!
    │1998-08-31│   2    │         │   2    │2020-03 │2020-04 │  51  │ ← JR2!
    │1998-09-30│   2    │         │   ...  │  ...   │  ...   │ ...  │
    │1998-10-01│   1    │         └────────┴────────┴────────┴──────┘
    └──────────┴────────┘
    
    Output: data/jm/jump_regime_segments.csv
    
    Results:
    • Total Segments: 89
      - JR0: 33 segments (avg 241 days each)
      - JR1: 48 segments (avg  28 days each)
      - JR2:  8 segments (avg  41 days each)
    
    JR2 Segments (The 8 Crisis Periods):
    ┌────┬────────────┬────────────┬──────────┬─────────────────────┐
    │ ID │   Start    │    End     │ Duration │   Crisis Event      │
    ├────┼────────────┼────────────┼──────────┼─────────────────────┤
    │ 1  │ 1998-08-31 │ 1998-09-30 │  30 days │ LTCM Collapse       │
    │ 2  │ 2002-07-24 │ 2002-08-21 │  28 days │ WorldCom/Enron      │
    │ 3  │ 2002-10-15 │ 2002-10-29 │  14 days │ Dot-com Bottom      │
    │ 4  │ 2008-09-18 │ 2009-01-06 │ 110 days │ Lehman Collapse 💥  │
    │ 5  │ 2009-02-24 │ 2009-04-07 │  42 days │ 2008 Aftermath      │
    │ 6  │ 2011-08-09 │ 2011-09-08 │  30 days │ US Debt Downgrade   │
    │ 7  │ 2020-03-04 │ 2020-04-24 │  51 days │ COVID-19 Crash 💥   │
    │ 8  │ 2025-04-09 │ 2025-05-02 │  23 days │ Future Period       │
    └────┴────────────┴────────────┴──────────┴─────────────────────┘


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│              STEP 6: TASK CONSTRUCTION (Segment-Based Approach)              │
│               Script: scripts/jm/create_segment_based_tasks.py               │
└─────────────────────────────────────────────────────────────────────────────┘

    **Why Segment-Based Instead of Sliding Window?**
    
    Sliding window FAILED: Only 2 JR2 tasks, missed 2008 Lehman! ❌
    Segment-based SUCCESS: 5 JR2 tasks, includes all major crises ✅
    
    Parameters:
    • Min Task Length: 30 days (minimum viable)
    • Target Task Length: 40 days (for padding short segments)
    • Direct Segment Mapping: One task per segment (no overlap)
    
    
    SEGMENT-BASED APPROACH:
    
    For each JR2 segment:
    
    IF segment ≥ 40 days:
        ✓ Use segment as-is (100% purity)
        Example: 2008 Lehman (110d) → Task (75d, pure JR2)
    
    ELIF 30 ≤ segment < 40 days:
        ✓ Pad with surrounding days to reach 40d
        Example: 1998 LTCM (30d) → Task (27d, 75% purity)
    
    ELSE segment < 30 days:
        ✗ Skip (too short)
        Example: 2002 WorldCom (28d) → SKIPPED
    
    
    SEGMENT MAPPING RESULTS:
    
    8 JR2 Segments → 5 JR2 Tasks Created:
    
    ┌────┬─────────────────────┬──────────┬─────────┬────────────┐
    │ ID │   Crisis Event      │ Segment  │  Task   │   Result   │
    ├────┼─────────────────────┼──────────┼─────────┼────────────┤
    │ 1  │ 1998 LTCM           │  30 days │  27d    │ ✓ Task 44  │
    │ 2  │ 2002 WorldCom       │  28 days │   -     │ ✗ Skipped  │
    │ 3  │ 2002 Dot-com        │  14 days │   -     │ ✗ Skipped  │
    │ 4  │ 2008 Lehman 💥      │ 110 days │  75d    │ ✓ Task 45  │
    │ 5  │ 2009 Aftermath      │  42 days │  30d    │ ✓ Task 46  │
    │ 6  │ 2011 US Downgrade   │  30 days │  27d    │ ✓ Task 47  │
    │ 7  │ 2020 COVID-19 💥    │  51 days │  36d    │ ✓ Task 48  │
    │ 8  │ 2025 Future         │  23 days │   -     │ ✗ Skipped  │
    └────┴─────────────────────┴──────────┴─────────┴────────────┘
    
    Success Rate: 5/8 segments captured (62.5%)
    Most Important: 2008 Lehman & 2020 COVID BOTH INCLUDED! ⭐


                            ↓ ↓ ↓
    
    EXAMPLE: 2008 LEHMAN CRISIS (Task 45)
    
    Segment: 2008-09-18 to 2009-01-06 (110 days)
    
    ┌─────────────────────────────────────────────────────────────┐
    │         2008-09-18                    2009-01-06            │
    │              ├──────────110 days JR2──────────┤             │
    │              JR2 JR2 JR2 ... JR2 JR2 JR2 JR2 JR2           │
    └─────────────────────────────────────────────────────────────┘
    
    Result: Segment ≥ 40 days → Use as-is!
    
    Task Created:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Task 45: 2008-09-18 to 2009-01-06 (75 days used)              │
    │  Purity: 100% (pure JR2, no padding needed!)                    │
    │  Jump Frequency: 40.8% (HIGHEST in dataset!)                    │
    │  Avg Jump Size: 5.80%                                            │
    │  Max VIX: 80.9                                                   │
    └──────────────────────────────────────────────────────────────────┘
    
    Key Jumps Captured:
    • Sep 29: -8.81% (Congress rejects bailout)
    • Oct 15: -7.87% (market panic)
    • Oct 28: +11.58% (coordinated central bank action)
    • Nov 20: -6.71% (bankruptcy fears)
    • Dec 01: +8.93% (stimulus hopes)
    
    
    EXAMPLE: 1998 LTCM (Task 44) - Short Segment
    
    Segment: 1998-08-31 to 1998-09-30 (30 days)
    
    Problem: 30 days < 40 day target → Need padding
    
    Solution: Pad 5 days before + 5 days after
    
    Result:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Task 44: 1998-08-26 to 1998-10-05 (27 days actual)            │
    │  Purity: 75% (30d JR2 / 40d target)                             │
    │  Jump Frequency: 17.9%                                           │
    │  Avg Jump Size: 4.64%                                            │
    └──────────────────────────────────────────────────────────────────┘


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                  STEP 7: FEATURE ENGINEERING PER TASK                        │
│               Script: scripts/jm/create_segment_based_tasks.py               │
└─────────────────────────────────────────────────────────────────────────────┘

    For EACH task window, calculate:
    
    1. Jump Frequency
       = (# jumps in window) / (# days in window)
       Example: 11 jumps in 30 days = 0.37 (37%)
    
    2. Average Jump Size
       = Mean(|return| for all jump days)
       Example: (3.5% + 5.2% + 11.98%) / 3 = 6.9%
    
    3. Jump Clustering
       = How close together are jumps?
       Formula: Std(days between jumps) / Mean(days between jumps)
       Example: Jumps on day 1, 2, 5, 8 → High clustering
    
    4. Negative Jump Ratio
       = (# negative jumps) / (# total jumps)
       Example: 6 negative, 5 positive → 0.55 (55% negative)
    
    5. Max VIX
       = Highest VIX level in window
    
    6. Average VIX
       = Mean VIX level in window
    
    
    Example Task (Task 48 - COVID Crisis):
    ┌─────────────────────────┬──────────────────────┐
    │       Feature           │        Value         │
    ├─────────────────────────┼──────────────────────┤
    │ Task ID                 │         48           │
    │ Regime                  │          2 (JR2)     │
    │ Regime Purity           │       1.00 (100%)    │
    │ Start Date              │    2020-03-04        │
    │ End Date                │    2020-04-24        │
    │ Length                  │      36 days         │
    │ Segment Duration        │      51 days         │
    │ Jump Frequency          │       0.378 (37.8%)  │
    │ Avg Jump Size           │       6.82%          │
    │ Jump Clustering         │      -0.041          │
    │ Negative Jump Ratio     │       0.53 (53%)     │
    │ Max VIX                 │       82.7           │
    │ Avg VIX                 │       55.3           │
    │ Source                  │   segment_exact      │
    └─────────────────────────┴──────────────────────┘


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│               STEP 8: TRAIN/VAL/TEST SPLIT (Temporal)                        │
│               Script: scripts/jm/create_segment_based_tasks.py               │
└─────────────────────────────────────────────────────────────────────────────┘

    Strategy: Chronological split (avoid data leakage)
    
    Total Tasks: 49 (113% more than sliding window!)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                         Timeline                                │
    ├──────────────────────┬─────────────┬──────────────────────────┤
    │   TRAIN (51%)        │  VAL (12%)  │      TEST (37%)          │
    │   25 tasks           │   6 tasks   │      18 tasks            │
    │  1991-2009           │ 2010-2011   │    2011-2025             │
    └──────────────────────┴─────────────┴──────────────────────────┘
    
    Train Tasks (25):
    • JR0: 14 tasks
    • JR1:  8 tasks
    • JR2:  3 tasks ✅ (1998 LTCM, 2008 Lehman, 2009 Aftermath)
    
    Validation Tasks (6):
    • JR0:  3 tasks
    • JR1:  2 tasks
    • JR2:  1 task ✅ (2011 US Debt Downgrade)
    
    Test Tasks (18):
    • JR0: 12 tasks
    • JR1:  5 tasks
    • JR2:  1 task ✅ (2020 COVID-19 - Holdout Crisis!)
    
    Output Files:
    • data/jm/tasks_metadata.json (49 tasks, 5 JR2)
    • data/jm/task_split_indices.json (train/val/test indices)


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                   STEP 9: FINAL TASK STRUCTURE                               │
│                      Ready for MAML Training                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    Task Distribution by Regime:
    
    ┌─────────┬────────────┬──────────────────────────────────────┐
    │ Regime  │ # Tasks    │           Crisis Events              │
    ├─────────┼────────────┼──────────────────────────────────────┤
    │ JR0     │    29      │ Calm market periods (no major jumps) │
    │ JR1     │    15      │ Moderate volatility periods          │
    │ JR2     │     5      │ 5 Major Crises Captured! ⭐          │
    └─────────┴────────────┴──────────────────────────────────────┘
    
    
    5 JR2 Crisis Tasks (150% Improvement!):
    
    ✅ Task 44: 1998 LTCM Collapse (75% purity, 17.9% jump freq)
    ✅ Task 45: 2008 Lehman Brothers 💥 (100% purity, 40.8% jump freq)
    ✅ Task 46: 2009 Financial Aftermath (100% purity, 19.4% jump freq)
    ✅ Task 47: 2011 US Debt Downgrade (75% purity, 21.4% jump freq)
    ✅ Task 48: 2020 COVID-19 Crash 💥 (100% purity, 37.8% jump freq)
    
    Why Segment-Based Approach Works Better:
    
    ✓ Direct mapping from segments (no stride issues)
    ✓ Captures variable-length crises (14d to 110d)
    ✓ High purity (75-100% vs 40-77% with sliding window)
    ✓ Includes 2008 Lehman (was MISSING in sliding window!)
    ✓ 5 JR2 tasks vs 2 (150% improvement)
    
    
    FINAL OUTPUT STRUCTURE:
    
    Each task contains:
    ┌──────────────────────────────────────────────────────────┐
    │ Task Metadata (tasks_metadata.json)                      │
    ├──────────────────────────────────────────────────────────┤
    │ • task_id: 18                                            │
    │ • regime: 2 (JR2)                                        │
    │ • regime_purity: 0.77                                    │
    │ • start_date: "2020-02-24"                               │
    │ • end_date: "2020-04-04"                                 │
    │ • length: 30 days                                        │
    │ • start_idx: 7545                                        │
    │ • end_idx: 7575                                          │
    │ • jump_frequency: 0.37                                   │
    │ • avg_jump_size: 0.0238                                  │
    │ • jump_clustering: -0.041                                │
    │ • negative_jump_ratio: 0.53                              │
    │ • max_vix: 82.7                                          │
    │ • avg_vix: 55.3                                          │
    └──────────────────────────────────────────────────────────┘
    
    Data Access:
    • Use start_idx/end_idx to slice main dataframe
    • Extract features: returns, VIX, volatility, jump metrics
    • Split into support (40d) and query (20d) sets
    • Feed to MAML for meta-learning


                            ↓ ↓ ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                       NEXT: MAML TRAINING (TODO)                             │
│                   Script: scripts/jm/train_maml_jm.py                        │
└─────────────────────────────────────────────────────────────────────────────┘

    For each training iteration:
    
    1. Sample batch of tasks (e.g., 4 tasks from train set)
    
    2. For each task:
       • Load support set (40 days) → train task-specific model
       • Load query set (20 days) → test adapted model
       • Calculate loss on query set
    
    3. Meta-update: Update meta-parameters to minimize query loss
    
    4. Repeat for validation and test sets
    
    Goal: Model learns to quickly adapt to new market regimes
          using just 40 days of data (support set)
    
    Expected: JR2 tasks provide crisis adaptation ability
```

---

## 📊 Summary Statistics

### Data Flow
```
Raw Data (9,020 days)
    ↓
Jump Detection (126 jumps @ 3.5% threshold)
    ↓
Regime Labels (9,020 daily labels)
    ↓
Regime Segments (75 segments: 31 JR0, 36 JR1, 8 JR2)
    ↓
Segment-Based Tasks (49 tasks from direct segment mapping)
    ↓
Train/Val/Test Split (25/6/18 tasks)
    ↓
MAML Training (READY ✅)
```

### Key Parameters
- **Jump Threshold**: 3.5% (|daily return|)
- **Intensity Window**: 20 days
- **JR0 Threshold**: < 0.05 intensity
- **JR1 Threshold**: 0.05 - 0.15 intensity
- **JR2 Threshold**: ≥ 0.15 intensity
- **Min Task Length**: 30 days
- **Target Task Length**: 40 days
- **Approach**: Segment-based (direct mapping, no stride)

### Final Results: 5 JR2 Tasks ✅
```
Task 44: 1998 LTCM Collapse      (27d, 75% purity, 17.9% jump freq)
Task 45: 2008 Lehman Brothers 💥  (75d, 100% purity, 40.8% jump freq)
Task 46: 2009 Financial Aftermath (30d, 100% purity, 19.4% jump freq)
Task 47: 2011 US Debt Downgrade  (27d, 75% purity, 21.4% jump freq)
Task 48: 2020 COVID-19 Crash 💥   (36d, 100% purity, 37.8% jump freq)
```

### Why Segment-Based Approach?
**Old Sliding Window Approach (FAILED):**
- 60-day window, 20-day stride, 40% purity
- Result: Only 2 JR2 tasks
- **Missing 2008 Lehman Crisis!** ❌

**New Segment-Based Approach (SUCCESS):**
- Direct mapping from segments
- Adaptive length (30-110 days)
- Result: 5 JR2 tasks
- **Includes 2008 Lehman!** ✅
- 150% improvement

---

**Created**: November 2025  
**Purpose**: Visual explanation of Jump Model → MAML task pipeline
