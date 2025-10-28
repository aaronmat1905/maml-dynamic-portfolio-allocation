# Data cleaning pipeline 
# src/data_cleaning.py (or similar name)

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def load_raw_data(sp500_path: str = "data/raw/sp500_raw_full.csv",
                  vix_path: str = "data/raw/vix_raw_full.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw S&P500 and VIX CSVs.

    The provided CSVs include a couple of extra header rows (Ticker / Date rows). We read the
    first row as the column names and skip the next two metadata rows before parsing the data.

    Returns DataFrames indexed by a DatetimeIndex (index name: Date).
    """
    # skiprows to remove the 'Ticker,...' and 'Date,...' extra lines after the header row
    sp500 = pd.read_csv(sp500_path, header=0, skiprows=[1, 2])
    vix = pd.read_csv(vix_path, header=0, skiprows=[1, 2])

    # The first column is labelled 'Price' in the files and contains the date strings
    for df in (sp500, vix):
        if 'Price' in df.columns:
            df.rename(columns={'Price': 'Date'}, inplace=True)

    # Parse dates and set index
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    vix['Date'] = pd.to_datetime(vix['Date'])
    sp500.set_index('Date', inplace=True)
    vix.set_index('Date', inplace=True)

    return sp500, vix


def _find_adj_close_column(df: pd.DataFrame) -> Optional[str]:
    # prefer columns that contain 'Adj' (case-insensitive)
    for col in df.columns:
        if 'adj' in col.lower():
            return col
    # fallback: first numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def clean_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Clean a price DataFrame.

    Steps:
    - Coerce numeric columns to numeric types
    - Drop exact duplicate rows (by index and values)
    - Drop rows where the adjusted close (or primary numeric) is NaN
    - Ensure index is sorted DatetimeIndex

    Returns a cleaned copy of the DataFrame.
    """
    print(f"Cleaning {name}...")
    df = df.copy()

    # Coerce numeric columns where possible
    for col in df.columns:
        # try to convert (ignore errors so non-numeric columns remain)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop exact duplicate rows
    df = df[~df.index.duplicated(keep='first')]

    # Identify primary price column (adjusted close preferred)
    adj_col = _find_adj_close_column(df)
    if adj_col is None:
        # nothing sensible to do
        print(f"Warning: no numeric/Adj Close column found for {name}")
        # drop rows that are all-NaN
        df = df.dropna(how='all')
        df.sort_index(inplace=True)
        return df

    # Drop rows where adjusted close is NaN
    df = df[~df[adj_col].isna()]

    # Sort by date
    df.sort_index(inplace=True)

    return df


def verify_price_ranges(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Check for obvious errors and remove invalid rows.

    Current checks:
    - adjusted close must be > 0 and not absurdly large
    - VIX values must be within a reasonable range (0, 500)

    Returns DataFrame with invalid rows removed.
    """
    df = df.copy()
    adj_col = _find_adj_close_column(df)
    if adj_col is None:
        return df

    invalid_mask = (df[adj_col] <= 0) | (df[adj_col].abs() > 1e6)

    # if this appears to be VIX data, apply a tighter upper bound
    if 'vix' in name.lower() or 'vix' in adj_col.lower():
        invalid_mask |= (df[adj_col] > 500)

    if invalid_mask.any():
        n = invalid_mask.sum()
        print(f"verify_price_ranges: dropping {n} invalid rows in {name} based on {adj_col}")
        df = df.loc[~invalid_mask]

    return df


def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns for the primary adjusted close column.

    Adds a new column named '<col>_log_return' and returns the augmented DataFrame.
    """
    df = df.copy()
    adj_col = _find_adj_close_column(df)
    if adj_col is None:
        return df

    ret_col = f"{adj_col}_log_return"
    # Use natural log differences
    df[ret_col] = np.log(df[adj_col]).diff()

    return df


def compute_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Helper: compute rolling standard deviation (no annualization)."""
    # Use min_periods=1 so NaNs inside the window are ignored when computing the std.
    # But to match the test expectation we only emit a value once the window index
    # reaches the full window size (i+1 >= window). Therefore compute with
    # min_periods=1 and then mask the first `window-1` entries to NaN.
    rolled = series.rolling(window=window, min_periods=1).std()
    if window > 1:
        rolled.iloc[: window - 1] = np.nan
    return rolled


def merge_data(sp500: pd.DataFrame, vix: pd.DataFrame, how: str = 'inner') -> pd.DataFrame:
    """Merge S&P500 and VIX on the Date index.

    By default performs an inner join (only dates present in both). Returns the merged DataFrame.
    """
    merged = pd.merge(sp500, vix, left_index=True, right_index=True, how=how, suffixes=('_SP500', '_VIX'))
    # Sort by date
    merged.sort_index(inplace=True)
    return merged


def save_clean_data(df: pd.DataFrame, filename: str):
    """Save DataFrame to `data/processed/filename` (CSV)."""
    os.makedirs('data/processed', exist_ok=True)
    path = os.path.join('data', 'processed', filename)
    df.to_csv(path, index=True)


def run_pipeline(save_filename: str = 'sp500_vix_merged_clean.csv') -> pd.DataFrame:
    """Execute the minimal data cleaning pipeline and return the merged DataFrame."""
    sp500, vix = load_raw_data()
    sp500 = clean_data(sp500, 'SP500')
    vix = clean_data(vix, 'VIX')
    sp500 = verify_price_ranges(sp500, 'SP500')
    vix = verify_price_ranges(vix, 'VIX')
    sp500 = calculate_log_returns(sp500)
    vix = calculate_log_returns(vix)
    merged = merge_data(sp500, vix)
    save_clean_data(merged, save_filename)
    print(f"Saved cleaned merged data to data/processed/{save_filename}")
    return merged


if __name__ == "__main__":
    # Run the pipeline (when invoked directly)
    run_pipeline()