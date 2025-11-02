"""Feature engineering utilities.

This module computes a small set of commonly used time-series features from the
cleaned, merged S&P500 + VIX DataFrame produced by the data cleaning pipeline.

Features produced (10 total):
1. SP500 log return (already present in cleaned data)
2. VIX log return (already present)
3. rolling mean of SP500 log returns (5-day)
4. rolling mean of SP500 log returns (20-day)
5. rolling std (volatility) of SP500 log returns (20-day)
6. momentum (10-day cumulative log return)
7. RSI (14-day) on SP500 log returns
8. ATR (14-day) — average true range from SP500 high/low/close
9. volume_pct_change (1-day) for SP500 volume
10. high_low_range (daily high-low normalized by close)

Also provides normalization, saving, and simple visualization helpers, and loading of saved scaler and normalized features.
"""

from typing import Iterable, List
import os

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pickle


def load_merged(path: str = os.path.join("data", "processed", "sp500_vix_merged_clean.csv")) -> pd.DataFrame:
    """Load the cleaned merged CSV and return a DataFrame indexed by Date."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) over a window on a return series.

    Uses the Wilder smoothing (exponential-like) via rolling mean on gains/losses.
    Returns a Series aligned with the input index.
    """
    delta = series.fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use simple rolling mean of gains/losses for RSI calculation
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (ATR).

    True range = max(high - low, abs(high - prev_close), abs(low - prev_close)).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the feature matrix from merged DataFrame.

    Expects columns with prefixes `SP500_` and `VIX_` similar to the cleaned CSV.
    Returns a DataFrame indexed by Date containing the features.
    """
    out = pd.DataFrame(index=df.index)

    # Basic returns (these columns exist from data_cleaning)
    if "SP500_Adj Close_log_return" in df.columns:
        out["sp_log_return"] = df["SP500_Adj Close_log_return"]
    else:
        out["sp_log_return"] = np.log(df["SP500_Adj Close"]).diff()

    if "VIX_Adj Close_log_return" in df.columns:
        out["vix_log_return"] = df["VIX_Adj Close_log_return"]
    else:
        out["vix_log_return"] = np.log(df["VIX_Adj Close"]).diff()

    # VIX Regime Features (forced selling detection)
    # Based on institutional insights: VIX > 40 = forced selling, predictable recovery
    if "VIX_Close" in df.columns:
        vix_close = df["VIX_Close"]
    elif "VIX_Adj Close" in df.columns:
        vix_close = df["VIX_Adj Close"]
    else:
        vix_close = None
    
    if vix_close is not None:
        out["vix_above_40"] = (vix_close > 40).astype(int)  # Panic/forced selling regime
        out["vix_above_30"] = (vix_close > 30).astype(int)  # Stressed regime
        out["vix_above_20"] = (vix_close > 20).astype(int)  # Elevated regime
        out["vix_spike"] = (vix_close.pct_change() > 0.10).astype(int)  # 10%+ daily spike
        out["vix_level"] = vix_close  # Raw VIX level for reference
    else:
        # Fallback if VIX_Close not found
        out["vix_above_40"] = 0
        out["vix_above_30"] = 0
        out["vix_above_20"] = 0
        out["vix_spike"] = 0
        out["vix_level"] = np.nan

    # Rolling means
    out["sp_return_ma5"] = out["sp_log_return"].rolling(window=5).mean()
    out["sp_return_ma20"] = out["sp_log_return"].rolling(window=20).mean()

    # Rolling volatility
    out["sp_vol_20"] = out["sp_log_return"].rolling(window=20).std()

    # Momentum: cumulative log return over 10 days (sum of log returns)
    out["sp_momentum_10"] = out["sp_log_return"].rolling(window=10).sum()

    # RSI on returns
    out["sp_rsi_14"] = _rsi(out["sp_log_return"], window=14)

    # ATR
    # prefer columns with SP500_High, SP500_Low, SP500_Close
    if all(c in df.columns for c in ["SP500_High", "SP500_Low", "SP500_Close"]):
        out["sp_atr_14"] = _atr(df["SP500_High"], df["SP500_Low"], df["SP500_Close"], window=14)
    else:
        out["sp_atr_14"] = np.nan

    # Volume change (pct)
    if "SP500_Volume" in df.columns:
        out["sp_volume_pct_change_1"] = df["SP500_Volume"].pct_change()
    else:
        out["sp_volume_pct_change_1"] = np.nan

    # High-low range normalized by close
    if all(c in df.columns for c in ["SP500_High", "SP500_Low", "SP500_Adj Close"]):
        out["sp_high_low_range"] = (df["SP500_High"] - df["SP500_Low"]) / df["SP500_Adj Close"]
    else:
        out["sp_high_low_range"] = np.nan

    # Keep same dtype and sorting
    out = out.sort_index()
    return out


def normalize_features(features: pd.DataFrame, method: str = "zscore") -> tuple:
    """Normalize feature matrix and return (normalized_df, scaler_dict).

    Supported methods: 'zscore' (mean=0,std=1) and 'minmax' (0-1).
    The returned scaler_dict contains the method and parameter mappings
    required to reproduce the normalization (e.g. mean/std or min/max).
    """
    features = features.copy()
    # replace infinite values with NaN to avoid contaminating summary stats
    features = features.replace([np.inf, -np.inf], np.nan)
    numeric = features.select_dtypes(include=["number"]).columns
    scaler = {"method": method, "params": {}}
    if method == "zscore":
        mu = features[numeric].mean()
        sigma = features[numeric].std()
        features[numeric] = (features[numeric] - mu) / sigma.replace(0, np.nan)
        scaler["params"]["mean"] = mu.to_dict()
        scaler["params"]["std"] = sigma.to_dict()
    elif method == "minmax":
        lo = features[numeric].min()
        hi = features[numeric].max()
        denom = (hi - lo).replace(0, np.nan)
        features[numeric] = (features[numeric] - lo) / denom
        scaler["params"]["min"] = lo.to_dict()
        scaler["params"]["max"] = hi.to_dict()
    else:
        raise ValueError(f"unknown normalization method: {method}")
    return features, scaler


def save_features(features: pd.DataFrame, filename: str = "features.csv") -> str:
    """Save feature matrix to data/processed and return path."""
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    features.to_csv(path, index=True)
    return path

def load_feature_scaler(path: str = os.path.join("data", "processed", "feature_scaler.pkl")) -> dict:
    """Load and return the saved feature scaler dict (if present).

    Returns the scaler dict (example: {"method":"zscore","params":{...}}) or
    raises FileNotFoundError if the file is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"scaler not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_normalized_features(path: str = os.path.join("data", "processed", "features_with_scaler.csv")) -> pd.DataFrame:
    """Load the canonical normalized features CSV (features_with_scaler.csv).

    This function returns a DataFrame indexed by Date. If the file is missing
    it raises FileNotFoundError.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"normalized features not found at {path}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def plot_features(features: pd.DataFrame, cols: Iterable[str], out_dir: str = "docs/figures") -> List[str]:
    """Create simple line plots for the requested columns and save PNGs.

    Returns a list of saved file paths.
    """
    # import here to avoid requiring matplotlib for non-plotting usage
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for c in cols:
        if c not in features.columns:
            continue
        plt.figure(figsize=(10, 3))
        plt.plot(features.index, features[c], lw=1)
        plt.title(c)
        plt.xlabel("Date")
        plt.tight_layout()
        p = os.path.join(out_dir, f"feature_{c}.png")
        plt.savefig(p)
        plt.close()
        saved.append(p)
    return saved


def prepare_features_for_modeling(
    *,
    impute: bool = True,
    add_volume_log: bool = True,
    drop_initial: int | None = None,
    imputed_out_csv: str = os.path.join("data", "processed", "features_imputed.csv"),
    imputer_json: str = os.path.join("data", "processed", "imputer_medians.json"),
) -> tuple[pd.DataFrame, dict]:
    """Prepare a dataset ready for modeling (HMM).

    Steps performed:
    - Load canonical normalized features from `features_with_scaler.csv`.
    - Optionally compute a robust volume feature `sp_volume_log_return`
      from the cleaned merged CSV (`sp500_vix_merged_clean.csv`) and add it as
      an extra column (doesn't replace the original features).
    - Replace inf/-inf with NaN, then optionally median-impute numeric
      columns and save medians to `imputer_json`.
    - Add `imputed_<col>` boolean mask columns indicating which values were
      imputed (True if the original value was NaN).

    Returns (df_imputed, medians_dict). The original `features_with_scaler.csv`
    is left unchanged.
    """
    # load canonical normalized features
    df = load_normalized_features()

    # optionally compute robust volume log-return from raw merged data
    if add_volume_log:
        try:
            merged = load_merged()
            if "SP500_Volume" in merged.columns:
                vol = merged["SP500_Volume"].fillna(0)
                # log(Vol + 1).diff() is robust to zero values
                vol_log = np.log(vol + 1).diff()
                # align to df index (merge by Date index)
                vol_log = vol_log.reindex(df.index)
                df["sp_volume_log_return"] = vol_log
            else:
                # no raw volume available; leave the column absent
                pass
        except Exception:
            # if loading merged data fails, continue without the extra feature
            pass

    # drop initial rows if requested (preserves user's choice)
    if drop_initial is not None and isinstance(drop_initial, int) and drop_initial > 0:
        df = df.iloc[drop_initial:]

    # sanitize infs
    df = df.replace([np.inf, -np.inf], np.nan)

    medians = {}
    if impute:
        numeric = df.select_dtypes(include=["number"]).columns
        med = df[numeric].median()
        medians = med.to_dict()
        # fill NaNs with medians
        df_imputed = df.fillna(med)
        # add mask columns for imputation (True where we replaced a NaN)
        for c in numeric:
            mask_name = f"imputed_{c}"
            df_imputed[mask_name] = df[c].isna()
    else:
        df_imputed = df.copy()

    # ensure output directory exists and save artifacts
    out_dir = os.path.dirname(imputed_out_csv) or os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    df_imputed.to_csv(imputed_out_csv, index=True)
    with open(imputer_json, "w", encoding="utf-8") as jf:
        json.dump(medians, jf, indent=2)

    return df_imputed, medians


def run_feature_pipeline(save_filename: str = "features.csv", normalize: bool = True) -> pd.DataFrame:
    """Load merged cleaned data, compute features, normalize (optional), save and return features."""
    merged = load_merged()
    features = compute_features(merged)
    scaler = None
    if normalize:
        features, scaler = normalize_features(features, method="zscore")
        # save scaler for reproducibility
        out_dir = os.path.join("data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        scaler_path = os.path.join(out_dir, "feature_scaler.pkl")
        try:
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            print(f"Saved feature scaler to {scaler_path}")
            # Also save a JSON copy with metadata for portability and
            # long-term reproducibility (JSON is safer/portable than pickle).
            json_meta = {
                "method": scaler.get("method"),
                "params": scaler.get("params"),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "python_version": sys.version.split()[0],
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "features": list(features.columns),
                "notes": "Do NOT load the pickle from untrusted sources. Use this JSON for portability."
            }
            # sanitize params: replace NaN/Infinity with None for strict JSON
            def _sanitize(val):
                try:
                    if val is None:
                        return None
                    # handle dicts recursively
                    if isinstance(val, dict):
                        return {k: _sanitize(v) for k, v in val.items()}
                    if isinstance(val, (list, tuple)):
                        return [_sanitize(v) for v in val]
                    # numeric checks
                    if isinstance(val, float):
                        if np.isfinite(val):
                            return val
                        return None
                except Exception:
                    return None
                return val

            json_meta["params"] = _sanitize(json_meta.get("params", {}))
            json_path = os.path.join(out_dir, "feature_scaler.json")
            try:
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(json_meta, jf, indent=2)
                print(f"Saved feature scaler metadata to {json_path}")
            except Exception as je:
                print(f"Warning: failed to save scaler JSON: {je}")
        except Exception as e:
            print(f"Warning: failed to save scaler: {e}")

    path = save_features(features, filename=save_filename)
    print(f"Saved features to {path}")
    return features


if __name__ == "__main__":
    # Run pipeline when executed directly
    run_feature_pipeline()
# Feature engineering pipeline 
