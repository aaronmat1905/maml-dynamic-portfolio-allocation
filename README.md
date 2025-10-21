# Model-Agnostic Meta-Learning for Dynamic Portfolio Allocation under Market Regime Uncertainty

This repository implements a Model-Agnostic Meta-Learning (MAML) framework for dynamic portfolio allocation that adapts to changing market regimes such as volatility spikes, bullish/bearish transitions, and structural shifts. The goal is to design allocation models capable of rapid adaptation with minimal retraining while maintaining high generalization performance across unseen market conditions.

---

## Abstract

Traditional portfolio allocation models often assume market stability or require extensive retraining to adjust to new regimes. This limitation leads to delayed reactions during regime shifts, reducing profitability and robustness.  
To address this, we employ **Model-Agnostic Meta-Learning (MAML)** to obtain a meta-initialization capable of quickly adapting to new market states using only a few gradient updates. By training across multiple simulated and historical market conditions, the model learns how to learn—achieving rapid fine-tuning under regime uncertainty.

Our experiments show that MAML-based portfolio models outperform static baselines and continual learning methods in both adaptation speed and overall return stability.

---

## Methodology

1. **Data Generation & Regime Identification**
   - Historical financial time series are segmented into regimes (e.g., bull, bear, high-volatility).
   - Each regime serves as a separate task for meta-training.

2. **Meta-Training (MAML)**
   - The MAML algorithm learns a meta-initialization that generalizes across market regimes.
   - Each task’s support and query sets simulate training and adaptation phases.

3. **Fine-Tuning & Adaptation**
   - When a new market regime is detected, the model adapts rapidly using a few gradient steps.
   - No full retraining is required, minimizing data and computational overhead.

4. **Evaluation Metrics**
   - Cumulative Return (CR)
   - Sharpe Ratio
   - Maximum Drawdown (MDD)
   - Adaptation Speed (Steps to Convergence)

---

## Architecture Overview

```

Market Data  →  Regime Segmentation  →  MAML Meta-Training  →  Meta-Initialization θ*
↓
New Regime → Fine-tuning → Adapted Portfolio Policy

```

---

## Results (Placeholder)

| Model | Regimes Tested | Sharpe Ratio | Adaptation Steps | Max Drawdown |
|--------|----------------|---------------|------------------|---------------|
| Static LSTM | 4 | 0.92 | N/A | -12.4% |
| Continual Learning | 4 | 1.03 | 15 | -10.2% |
| **MAML (Proposed)** | 4 | **1.27** | **4** | **-7.9%** |

---

## Directory Structure

```

project_root/
│
├── data/
│   ├── raw/
│   │   ├── sp500_raw.csv
│   │   └── vix_raw.csv
│   ├── cleaned/
│   ├── features/
│   ├── processed/
│   │   └── regime_segments.pkl
│   └── tasks/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_regime_visualization.ipynb
│   └── 04_task_validation.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_acquisition.py
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   │   └── task_dataset.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hmm_regime.py
│   │   └── portfolio_net.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── plotting.py
│   │   ├── experiment_logger.py
│   │   └── validations.py
│   │
│   └── __init__.py
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_task_dataset.py
│   └── test_models.py
│
├── configs/
│   ├── experiment_template.yaml
│   └── hyperparams_tracker.xlsx
│
├── docs/
│   ├── 01_data_pipeline.md
│   ├── 02_feature_engineering.md
│   ├── 03_hmm_regime_detection.md
│   └── 04_meta_learning_tasks.md
│
├── requirements.txt
├── README.md
└── setup_kanban_structure.bat


````

---

## Dependencies

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  
- scikit-learn  

To install dependencies:

```bash
pip install -r requirements.txt
````

## Team Members
1. **Aaron Thomas Mathew** [https://github.com/aaronmat1905](https://github.com/aaronmat1905)
2. **Aman Kumar Mishra** [https://github.com/Aman-K-Mishra](https://github.com/Aman-K-Mishra)
3. **Anirudh Krishnan** [https://github.com/aaronmat1905](https://github.com/aaronmat1905)
4. **Preetham VJ** [https://github.com/PreethamVJ](https://github.com/PreethamVJ)
