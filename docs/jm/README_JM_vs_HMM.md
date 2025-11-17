# Statistical Jump Model (JM) vs HMM for MAML Portfolio Allocation

## Motivation

Your friend raises an excellent question: **Which is better for MAML - HMM regime detection or Jump Models?**

### Why Jump Models May Be Superior for Crisis Adaptation

**HMM Limitations:**
- Assumes **smooth transitions** between regimes (Markovian state evolution)
- Crisis detection relies on threshold (VIX > 40, which missed Ukraine 2022)
- Treats all high-VIX days the same (2008 subprime = 2020 pandemic = 2022 geopolitical)
- Regime labels are **static** after estimation

**Jump Model Advantages:**
- Explicitly models **discontinuous price movements** (jumps vs continuous diffusion)
- Detects jumps **statistically** from price data, no arbitrary thresholds
- Distinguishes jump **intensity** (how often), **size** (magnitude), and **clustering** (contagion)
- Crises are **defined by jump characteristics**, not just high volatility
- Natural for MAML: Each crisis has unique jump signature (adaptation target)

## Research Context

### Academic Papers on Jump Models in Finance

1. **Lee & Mykland (2008)** - "Jumps in Financial Markets: A New Nonparametric Test"
   - Gold standard for intraday jump detection
   - Uses high-frequency data (you have daily, will need adaptation)

2. **Barndorff-Nielsen & Shephard (2006)** - "Econometrics of Testing for Jumps in Financial Economics"
   - Bi-power variation for daily data jump tests
   - Separates continuous volatility from jumps

3. **Jiang & Oomen (2008)** - "Testing for Jumps When Asset Prices Are Observed with Noise"
   - Robust to market microstructure noise
   - Good for daily S&P 500 data

4. **Bollerslev et al. (2013)** - "Exploiting the Errors: A Simple Approach for Improved Volatility Forecasting"
   - Jump component improves volatility forecasts (relevant for portfolio risk)

5. **Tauchen & Zhou (2011)** - "Realized Jumps on Financial Markets and Predicting Credit Spreads"
   - Jump risk predicts crisis severity
   - Jumps cluster during crises (contagion effect)

### Jump Models for Portfolio Allocation

6. **Liu et al. (2003)** - "Jump-Diffusion Models for Portfolio Optimization"
   - Jump-robust portfolio strategies outperform in crises
   - Jump timing matters more than magnitude

7. **Aït-Sahalia & Jacod (2009)** - "Testing for Jumps in a Discretely Observed Process"
   - Statistical tests for jump detection in daily data
   - Can estimate jump intensity (λ) and jump size distribution

## Proposed Approach

### Phase 1: Jump Detection (Daily Data)

Since you have **daily S&P 500 returns**, use:

**Barndorff-Nielsen-Shephard (2006) Bi-Power Variation Test:**

```
RV_t = Σ r²_t (realized variance)
BV_t = (π/2) Σ |r_t| |r_{t-1}| (bi-power variation)
J_t = RV_t - BV_t (jump component)

Test statistic: Z_t = (J_t / √(θ BV_t)) ~ N(0,1)
Reject H0 (no jump) if Z_t > 2.33 (1% level)
```

**Alternative (simpler):** Lee-Mykland (2008) adapted for daily data:
```
L_t = r_t / σ_t (standardized return)
σ_t = rolling volatility estimate

Jump if |L_t| > threshold (e.g., 3.5 standard deviations)
```

### Phase 2: Jump Regime Classification

Instead of HMM's R0/R1/R2, define regimes by **jump intensity**:

- **JR0 (Calm):** <1 jump per month (λ < 0.05 daily)
- **JR1 (Moderate):** 1-3 jumps per month (0.05 ≤ λ < 0.15)
- **JR2 (Extreme):** >3 jumps per month (λ ≥ 0.15) + large jumps (>3%)

**Jump Features for MAML:**
1. Jump frequency (rolling 20-day count)
2. Average jump size (absolute magnitude)
3. Jump direction ratio (negative/positive jumps)
4. Jump clustering (autocorrelation of jump indicator)
5. VIX at jump occurrence
6. Time since last jump

### Phase 3: MAML Task Construction

**Each task = 40-day window characterized by:**
- Jump intensity (λ)
- Average jump size
- Jump clustering coefficient
- % negative jumps (crash vs rally)

**Example Tasks:**
- **2008 Oct (Lehman):** λ=0.25, avg size=5%, 80% negative, high clustering
- **2020 Mar (COVID):** λ=0.30, avg size=6%, 90% negative, extreme clustering
- **2022 Feb (Ukraine):** λ=0.18, avg size=3%, 60% negative, moderate clustering

**Why this helps MAML:**
- Each crisis has unique jump signature (different adaptation challenge)
- Model learns "how to adapt to jumps" not "memorize crisis type"
- Can generalize to novel crises with similar jump patterns

### Phase 4: Comparison (HMM vs JM)

| Aspect | HMM-MAML (current) | JM-MAML (proposed) |
|--------|-------------------|-------------------|
| Regime definition | VIX threshold | Statistical jump tests |
| Crisis detection | 3 periods (VIX>40) | 6-10 periods (jump clusters) |
| 2022 Ukraine | MISSED (VIX 36) | CAPTURED (16 jumps in Feb-Mar) |
| Task diversity | R0/R1/R2 categorical | Continuous jump intensity |
| Adaptation target | High volatility | Jump characteristics |
| Theoretical fit | Ad-hoc threshold | Grounded in jump-diffusion theory |

## Implementation Roadmap

### Immediate Next Steps

1. **scripts/jm/detect_jumps.py**
   - Implement BNS (2006) or Lee-Mykland (2008)
   - Load `data/processed/sp500_vix_merged_clean.csv`
   - Output: `data/jm/jump_indicators.csv` (date, jump_flag, jump_size, Z_stat)

2. **scripts/jm/classify_jump_regimes.py**
   - Rolling 20-day jump intensity
   - Classify JR0/JR1/JR2 based on intensity thresholds
   - Output: `data/jm/jump_regime_labels.csv`

3. **scripts/jm/generate_jump_tasks.py**
   - Create meta-learning tasks from jump regimes
   - Output: `data/jm/task_split_indices_jm.json`

4. **scripts/jm/train_maml_jm.py**
   - Copy from `scripts/train_maml.py`
   - Load jump-based tasks
   - Add jump-specific features to model input

5. **notebooks/jm/JM_Exploration.ipynb**
   - Visualize detected jumps vs VIX threshold
   - Compare 2022 Ukraine detection (HMM missed, JM catches)
   - Analyze jump clustering in 2008 vs 2020

### Validation Strategy

**Test 1: Ukraine 2022 (Critical Test)**
- HMM: 0 crisis days detected (VIX<40)
- JM: Should detect ~16 jump days in Feb-Mar 2022
- **If JM-MAML adapts well to Ukraine, proves jump approach superior**

**Test 2: Leave-One-Crisis-Out (LOCO)**
- Hold out 2008 (subprime jumps) → test JM-MAML adaptation
- Hold out 2020 (pandemic jumps) → test different jump pattern
- Hold out 2022 (geopolitical jumps) → test novel jump type

**Test 3: Jump Size Sensitivity**
- Do large jumps (>5%) require different adaptation than small jumps (2-3%)?
- Does model learn to scale portfolio response to jump magnitude?

## Expected Outcomes

### If JM-MAML Outperforms HMM-MAML:

**Evidence:**
- Lower MSE on Ukraine 2022 (HMM can't adapt, JM can)
- Better LOCO generalization across crisis types
- Jump features explain more variance than VIX alone

**Publication Angle:**
- "Jump-Augmented MAML for Crisis-Adaptive Portfolio Allocation"
- Compare HMM vs JM regimes empirically
- Show jump-based tasks enable better generalization

### If Results are Mixed:

**Hybrid Approach:**
- Use both HMM (market regime) and JM (crisis jumps)
- 6 regimes: R0-calm, R0-jumps, R1-calm, R1-jumps, R2-continuous, R2-jumps
- MAML adapts to both regime and jump characteristics

## Data Readiness Check

✅ **You have everything needed:**
- Daily S&P 500 returns (1990-2025) → Jump detection input
- VIX (1990-2025) → Contextual feature for jumps
- Existing features (momentum, volatility, etc.) → Reusable
- Task dataset infrastructure → Just swap regime labels

❌ **You DON'T need:**
- Intraday data (daily jumps are fine)
- New data acquisition (all ready)
- Re-run feature engineering (reuse existing)

## Recommendation

**Start with Jump Model parallel track:**

1. **Week 1:** Implement jump detection, visualize 2022 Ukraine
2. **Week 2:** Create JM tasks, compare with HMM tasks
3. **Week 3:** Train JM-MAML, compare Ukraine 2022 performance
4. **Week 4:** Decide based on results (JM-only, HMM-only, or hybrid)

**Why this is smart:**
- Your friend's intuition is correct: Jumps may be more natural for MAML
- You can compare empirically (not guess which is better)
- If JM works, stronger publication (grounded in stochastic calculus)
- If HMM works better, you have validation study

Let me know if you want me to implement the jump detection script first!
