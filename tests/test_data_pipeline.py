import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.data.data_cleaning import (
	clean_data,
	calculate_log_returns,
	compute_rolling_std,
	merge_data,
)


def make_price_df(dates, prices, colname='Adj Close'):
	df = pd.DataFrame({colname: prices}, index=pd.to_datetime(dates))
	df.index.name = 'Date'
	return df


def test_clean_and_log_returns():
	# small example where prices are known
	dates = ['2020-01-01', '2020-01-02', '2020-01-03']
	prices = [100.0, 101.0, 102.0]
	df = make_price_df(dates, prices, colname='SP500_Adj Close')

	cleaned = clean_data(df, 'SP500')
	assert cleaned.shape[0] == 3

	with_returns = calculate_log_returns(cleaned)
	# manual expected log returns
	expected = np.log(pd.Series(prices)).diff()
	pdt.assert_series_equal(with_returns['SP500_Adj Close_log_return'].reset_index(drop=True),
							expected.reset_index(drop=True),
							check_names=False)


def test_merge_and_alignment():
	# create two small series with overlapping dates
	dates1 = ['2020-01-01', '2020-01-02', '2020-01-03']
	p1 = [100.0, 101.0, 102.0]
	dates2 = ['2020-01-02', '2020-01-03', '2020-01-04']
	p2 = [20.0, 21.0, 22.0]

	sp = make_price_df(dates1, p1, colname='SP500_Adj Close')
	vix = make_price_df(dates2, p2, colname='VIX_Adj Close')

	sp = calculate_log_returns(clean_data(sp, 'SP500'))
	vix = calculate_log_returns(clean_data(vix, 'VIX'))

	merged = merge_data(sp, vix, how='inner')
	# inner merge should keep only 2020-01-02 and 2020-01-03
	assert pd.to_datetime('2020-01-02') in merged.index
	assert pd.to_datetime('2020-01-03') in merged.index
	assert pd.to_datetime('2020-01-01') not in merged.index


def test_rolling_std_matches_manual():
	# rolling std should match manual computation for a simple series
	prices = [100.0, 102.0, 101.0, 103.0, 104.0]
	dates = pd.date_range('2020-01-01', periods=len(prices))
	# make sure we call .diff() on a pandas Series (np.log returns ndarray)
	df = pd.Series(pd.Series(np.log(prices)).diff().values, index=dates)

	window = 3
	by_func = compute_rolling_std(df, window=window)

	# manual: for each window compute std of the values
	manual = pd.Series(index=dates, dtype=float)
	for i in range(len(df)):
		if i + 1 < window:
			manual.iloc[i] = np.nan
		else:
			window_vals = df.iloc[i - window + 1:i + 1]
			manual.iloc[i] = window_vals.std()

	pdt.assert_series_equal(by_func, manual)

