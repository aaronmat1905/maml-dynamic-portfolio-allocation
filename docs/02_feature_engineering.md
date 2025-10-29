# Feature engineering — formulas and usage

This document explains the 10 time-series features computed by
`src/data/feature_engineering.py`, the parameter choices used in the
implementation, how missing values are handled, and where the generated
artifacts are saved in the repository.

Files produced by the pipeline (canonical paths)
- `data/processed/features_unnormalized.csv` — raw, un-normalized feature matrix
- `data/processed/features_with_scaler.csv` — normalized feature matrix (canonical)
- `data/processed/feature_scaler.pkl` — saved normalization parameters (dict)

Overview
--------
The features are computed from the cleaned, merged S&P500 + VIX DataFrame
(`data/processed/sp500_vix_merged_clean.csv`). Where possible the pipeline
uses columns created by the cleaning step (for example, pre-computed
log-returns). All features are aligned by the `Date` index and returned as a
DataFrame with the same index.

Feature list and formulas
-------------------------
Below are the 10 features (column names are given in code-friendly form and
match the output of `compute_features()`):

1. sp_log_return
	 - Formula: sp_log_return_t = log(SP500_AdjClose_t) - log(SP500_AdjClose_{t-1})
	 - Implementation note: if the cleaned file already contains
		 `SP500_Adj Close_log_return` that column is used directly.

2. vix_log_return
	 - Formula: vix_log_return_t = log(VIX_AdjClose_t) - log(VIX_AdjClose_{t-1})
	 - Implementation note: uses `VIX_Adj Close_log_return` when present.

3. sp_return_ma5
	 - Formula: 5-day rolling mean of `sp_log_return`.
	 - Implementation: rolling(window=5).mean(), default min_periods is the
		 window length (so the first 4 rows are NaN).

4. sp_return_ma20
	 - Formula: 20-day rolling mean of `sp_log_return`.

5. sp_vol_20
	 - Formula: 20-day rolling sample standard deviation of `sp_log_return`.
	 - Implementation: rolling(window=20).std(). This is an empirical volatility
		 estimate over the lookback window.

6. sp_momentum_10
	 - Formula: 10-day cumulative log return = sum_{i=0..9} sp_log_return_{t-i}
	 - Implementation: rolling(window=10).sum(). Interpreted as a short-term
		 momentum signal.

7. sp_rsi_14
	 - Formula (Wilder-style approximation used):
		 - delta_t = sp_log_return_t
		 - gain_t = max(delta_t, 0), loss_t = max(-delta_t, 0)
		 - avg_gain = rolling_mean(gain, window=14)
		 - avg_loss = rolling_mean(loss, window=14)
		 - RS = avg_gain / avg_loss
		 - RSI = 100 - 100 / (1 + RS)
	 - Implementation note: the code uses a simple rolling mean to approximate
		 Wilder smoothing. The first ~13 values are NaN.

8. sp_atr_14
	 - Formula: Average True Range (14-day)
		 - TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
		 - ATR_t = rolling_mean(TR, window=14)
	 - Implementation note: requires `SP500_High`, `SP500_Low`, and
		 `SP500_Close` to be present. If they are missing the column is NaN.

9. sp_volume_pct_change_1
	 - Formula: one-day percentage change in reported volume:
		 - sp_volume_pct_change_1_t = (Volume_t - Volume_{t-1}) / Volume_{t-1}
	 - Implementation note: computed via pandas .pct_change(); will be NaN for
		 the first row or when prior volume is zero or missing.

10. sp_high_low_range
		- Formula: (High_t - Low_t) / AdjClose_t
		- Implementation note: uses adjusted close in denominator to account for
			corporate actions (if available as `SP500_Adj Close`).

Handling of missing values
--------------------------
- Rolling features use pandas rolling with `min_periods` equal to the window
	size. This intentionally produces NaNs for the initial window where an
	estimate is not statistically meaningful.
- The pipeline does not impute these NaNs automatically. Downstream models
	should either handle NaNs (e.g. by masking) or apply an imputation step as
	appropriate for the model.

Normalization / reproducibility
--------------------------------
The pipeline supports two normalization methods in `normalize_features()`:

- zscore (default): (x - mean) / std — the pipeline computes column-wise
	mean and std and saves them to `data/processed/feature_scaler.pkl` as a
	serialized dict of the form:

	{
		"method": "zscore",
		"params": {
			"mean": {"sp_log_return": ..., ...},
			"std": {"sp_log_return": ..., ...}
		}
	}

- minmax: scales to [0, 1] and the saved params are `min` and `max`.

Use the saved scaler to reproduce normalization exactly across environments by
loading the dict with `src.data.feature_engineering.load_feature_scaler()` and
applying the inverse/forward transform logic (the module exposes
`normalize_features()` which returns both normalized DataFrame and a scaler
dict when called programmatically).

Quick usage examples
--------------------
In Python, from the project root:

```python
from src.data import feature_engineering as fe

# Load canonical normalized features (raises if missing)
df = fe.load_normalized_features()

# Load saved scaler
scaler = fe.load_feature_scaler()

# Recompute features from raw merged data and re-normalize (if needed)
features = fe.compute_features(fe.load_merged())
normed, scaler_dict = fe.normalize_features(features, method="zscore")

# Create plots saved to docs/figures
fe.plot_features(normed, cols=normed.columns[:6], out_dir="docs/figures")
```

Notes and provenance
--------------------
- The features above are intentionally simple and commonly used in
	time-series momentum and volatility-based strategies.
- Window choices (5, 10, 14, 20) are conventional defaults; feel free to
	experiment by changing the window parameters in `compute_features()`.
- The canonical normalized CSV in the repository is
	`data/processed/features_with_scaler.csv`. The scaler required to reproduce
	normalization is `data/processed/feature_scaler.pkl`.

Security & portability warning about the pickle
------------------------------------------------
The scaler is saved as a Python pickle (`feature_scaler.pkl`) for convenience,
but pickles are not secure when loaded from untrusted sources and can be
version-sensitive (different Python / pandas releases may behave differently
when unpickling). To mitigate this:

- A JSON copy of the scaler metadata is also saved at
	`data/processed/feature_scaler.json`. Prefer using the JSON file to inspect
	the saved parameters, and to re-implement normalization in other
	environments.
- Never load `feature_scaler.pkl` from an untrusted or external repository —
	pickles can execute arbitrary code on load. Only load the pickle if you
	trust the repository and the environment.
- Include Python and pandas version information when archiving data for
	long-term reproducibility.



Figures
-------
Below are example feature visualizations produced by the pipeline. The
images are stored in `docs/figures` (relative to this document). Each figure
is a single-series line plot of the normalized feature over the full time
range.

<figure>
	<img src="figures/feature_sp_log_return.png" alt="SP500 log return" style="max-width:100%;">
	<figcaption>Figure 1 — SP500 log return (normalized)</figcaption>
</figure>

<figure>
	<img src="figures/feature_vix_log_return.png" alt="VIX log return" style="max-width:100%;">
	<figcaption>Figure 2 — VIX log return (normalized)</figcaption>
</figure>

<figure>
	<img src="figures/feature_sp_return_ma5.png" alt="SP500 5-day mean" style="max-width:100%;">
	<figcaption>Figure 3 — SP500 5-day rolling mean of log returns (sp_return_ma5)</figcaption>
</figure>

<figure>
	<img src="figures/feature_sp_return_ma20.png" alt="SP500 20-day mean" style="max-width:100%;">
	<figcaption>Figure 4 — SP500 20-day rolling mean of log returns (sp_return_ma20)</figcaption>
</figure>

<figure>
	<img src="figures/feature_sp_vol_20.png" alt="SP500 20-day vol" style="max-width:100%;">
	<figcaption>Figure 5 — SP500 20-day rolling volatility (sp_vol_20)</figcaption>
</figure>

<figure>
	<img src="figures/feature_sp_momentum_10.png" alt="SP500 momentum 10" style="max-width:100%;">
	<figcaption>Figure 6 — SP500 10-day momentum (sp_momentum_10)</figcaption>
</figure>

