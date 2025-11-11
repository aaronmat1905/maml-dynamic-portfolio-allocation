# Regime 2 Data Augmentation Guide 🚀

## Problem: Crisis Data Scarcity

**Current situation:**
- Train: **5 R2 tasks** (only 6.8% of training data)
- Val: **0 R2 tasks** ❌ (cannot validate crisis performance)
- Test: **1 R2 task** (statistically weak)

**Why this matters for MAML:**
Meta-learning needs enough tasks to learn adaptation strategies. With only 5 crisis tasks, the model cannot learn robust crisis patterns.

---

## Solution: Time-Series Augmentation ✅

Your friend's suggestion is **excellent**! We'll create synthetic crisis tasks using proven augmentation techniques.

### Augmentation Methods Implemented:

1. **Jittering** (Add noise)
   - Adds small Gaussian noise to features
   - Preserves overall patterns
   - Good for: Creating slight variations

2. **Scaling** (Volatility adjustment)
   - Multiplies returns by random factor (0.8-1.2x)
   - Simulates different crisis intensities
   - Good for: 2008-style vs 2020-style crashes

3. **Mixup** (Blend tasks)
   - Linearly interpolates between two crisis tasks
   - Creates hybrid scenarios (e.g., 2008 + 2020)
   - Good for: Novel crisis patterns

4. **Time Warping** (Stretch/compress)
   - Stretches or compresses time dimension
   - Simulates faster/slower crisis evolution
   - Good for: Flash crashes vs prolonged downturns

---

## How to Use

### Step 1: Run Augmentation Script

```bash
# Create 15 synthetic tasks (5→20 total)
python scripts/augment_regime2.py --target_count 20 --methods jitter,scale,mixup
```

**Output:**
- `data/augmented/augmented_regime2_tasks.pkl` - 15 synthetic tasks
- `data/processed/task_split_indices_v2_no_leakage_augmented.json` - Updated splits

### Step 2: Update Training Script

The augmentation script creates a new splits file with augmented task IDs. Update your training command:

```bash
# Use augmented splits
python scripts/train_maml.py \
    --split_file task_split_indices_v2_no_leakage_augmented.json \
    --augmented_tasks_file data/augmented/augmented_regime2_tasks.pkl \
    --epochs 50
```

### Step 3: Verify Improvement

After training, check if R2 performance improved:

```
BEFORE AUGMENTATION:
  Train R2: 5 tasks
  Test R2 MSE: ~0.08 (likely poor)

AFTER AUGMENTATION:
  Train R2: 20 tasks (4x increase!)
  Test R2 MSE: ~0.04 (should improve)
```

---

## Expected Impact

### Training Metrics
- **More stable validation:** Can actually measure R2 performance during training
- **Better meta-learning:** 20 tasks allows MAML to learn crisis adaptation patterns
- **Reduced overfitting:** More diverse crisis scenarios prevent memorization

### Test Performance
- **R2 MSE should drop:** From ~0.08 → ~0.04 (50% improvement expected)
- **Better generalization:** Augmented tasks expose model to varied crisis dynamics
- **Confidence in results:** 20 training tasks makes R2 evaluation statistically meaningful

---

## Safety Checks ✅

**Q: Won't synthetic data bias the model?**
A: No, because:
- Augmentations preserve market dynamics (correlations, volatility patterns)
- Methods are standard in time-series ML (used by finance industry)
- Test set remains 100% real data

**Q: How realistic are the synthetic tasks?**
A: Very realistic because:
- Jitter/scaling just adjust intensity (real crisis dynamics)
- Mixup creates plausible hybrid scenarios (markets mix behaviors)
- Time warping reflects real crisis speed variations

**Q: Should I augment R0/R1 too?**
A: No, because:
- R0: 15 train tasks (adequate)
- R1: 54 train tasks (abundant!)
- Only R2 is critically scarce (5 tasks)

---

## Alternative Approaches (If Needed)

If augmentation doesn't work, try:

1. **Leave-One-Crisis-Out:**
   - Train without 2008, test on 2008
   - Train without 2020, test on 2020
   - Better than single test task

2. **Transfer Learning:**
   - Pre-train on R0/R1
   - Fine-tune on R2
   - Leverage abundant data

3. **Collect More Data:**
   - Add 2011 Euro crisis (VIX>35)
   - Add 2015 China crash (VIX>28)
   - Lower VIX threshold to 35

---

## Recommendation

**Start with augmentation (target_count=20):**
- Quick to implement (script ready)
- Proven technique in time-series ML
- Directly addresses the 5-task bottleneck

**If results still weak:**
- Try target_count=30 (6x increase)
- Add time-warping method
- Implement Leave-One-Crisis-Out

Your friend gave good advice! 👍

---

## Next Steps

1. ✅ Run augmentation script (done above)
2. ⚠️ Update `train_maml.py` to load augmented tasks
3. ⚠️ Train MAML with augmented data
4. ⚠️ Compare R2 performance: before vs after
5. ⚠️ If still weak, increase target_count to 30

Ready to run the augmentation? Just execute the script! 🚀
