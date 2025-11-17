"""
Test different jump detection thresholds to find optimal balance.

This script tests various thresholds without modifying existing data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def test_threshold(returns, threshold_pct):
    """Test a specific threshold and return statistics."""
    abs_returns = returns.abs()
    jump_flag = (abs_returns > threshold_pct).astype(int)
    
    return {
        'threshold': f'{threshold_pct:.2%}',
        'total_jumps': jump_flag.sum(),
        'jump_frequency': f'{(jump_flag.sum() / len(returns)) * 100:.2f}%',
        'avg_days_between': f'{len(returns) / jump_flag.sum():.1f}' if jump_flag.sum() > 0 else 'N/A'
    }


def calculate_regime_stats(returns, threshold_pct):
    """Calculate jump intensity and estimate regime segments."""
    abs_returns = returns.abs()
    jump_flag = (abs_returns > threshold_pct).astype(int)
    
    # Calculate jump intensity (rolling 20-day window)
    jump_intensity = jump_flag.rolling(window=20, min_periods=5).mean()
    
    # Classify regimes
    regime_labels = pd.Series(0, index=returns.index)  # Default JR0
    regime_labels[jump_intensity >= 0.05] = 1  # JR1
    regime_labels[jump_intensity >= 0.15] = 2  # JR2
    
    # Count regime days
    regime_counts = regime_labels.value_counts().sort_index()
    
    # Estimate segments (consecutive days in same regime)
    regime_changes = (regime_labels != regime_labels.shift()).cumsum()
    segments = regime_labels.groupby(regime_changes).agg(['first', 'count'])
    segment_counts = segments.groupby('first').size()
    
    return {
        'jr0_days': regime_counts.get(0, 0),
        'jr1_days': regime_counts.get(1, 0),
        'jr2_days': regime_counts.get(2, 0),
        'jr0_segments': segment_counts.get(0, 0),
        'jr1_segments': segment_counts.get(1, 0),
        'jr2_segments': segment_counts.get(2, 0)
    }


def analyze_crisis_period(returns, threshold_pct, crisis_name, start_date, end_date):
    """Analyze jumps during a specific crisis period."""
    mask = (returns.index >= start_date) & (returns.index <= end_date)
    period_returns = returns[mask]
    
    if len(period_returns) == 0:
        return None
    
    abs_returns = period_returns.abs()
    jump_flag = (abs_returns > threshold_pct).astype(int)
    n_jumps = jump_flag.sum()
    n_days = len(period_returns)
    jump_freq = (n_jumps / n_days) * 100 if n_days > 0 else 0
    
    return {
        'crisis': crisis_name,
        'days': n_days,
        'jumps': n_jumps,
        'freq': f'{jump_freq:.1f}%'
    }


def main():
    # Load data
    data_path = Path('data/processed/sp500_vix_merged_clean.csv')
    print("="*80)
    print("THRESHOLD COMPARISON TEST")
    print("="*80)
    print(f"\nLoading data from {data_path}...")
    
    df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    df.columns = df.columns.str.lower()
    
    # Calculate returns
    if 'sp500_return' not in df.columns:
        df['sp500_return'] = df['sp500_close'].pct_change() if 'sp500_close' in df.columns else df['sp500_adj_close'].pct_change()
    
    returns = df['sp500_return'].dropna()
    
    print(f"Data range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"Total observations: {len(returns):,}\n")
    
    # Test different thresholds
    thresholds = [0.025, 0.030, 0.0325, 0.035, 0.040]
    
    print("="*80)
    print("THRESHOLD COMPARISON")
    print("="*80)
    print(f"{'Threshold':<12} {'Total Jumps':<15} {'Frequency':<15} {'Avg Days Between':<20}")
    print("-"*80)
    
    results = {}
    for threshold in thresholds:
        stats = test_threshold(returns, threshold)
        results[threshold] = stats
        print(f"{stats['threshold']:<12} {stats['total_jumps']:<15} {stats['jump_frequency']:<15} {stats['avg_days_between']:<20}")
    
    # Detailed analysis for 3.25% vs 3.5%
    print("\n" + "="*80)
    print("DETAILED COMPARISON: 3.25% vs 3.5%")
    print("="*80)
    
    for threshold in [0.0325, 0.035]:
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {threshold:.2%}")
        print(f"{'='*80}")
        
        # Regime statistics
        regime_stats = calculate_regime_stats(returns, threshold)
        
        print(f"\nRegime Days:")
        print(f"  JR0 (Calm):     {regime_stats['jr0_days']:,} days")
        print(f"  JR1 (Moderate): {regime_stats['jr1_days']:,} days")
        print(f"  JR2 (Extreme):  {regime_stats['jr2_days']:,} days")
        
        print(f"\nRegime Segments:")
        print(f"  JR0 segments: {regime_stats['jr0_segments']}")
        print(f"  JR1 segments: {regime_stats['jr1_segments']}")
        print(f"  JR2 segments: {regime_stats['jr2_segments']} ⭐")
        
        # Crisis period analysis
        crisis_periods = [
            ('2008 Financial Crisis', '2008-09-01', '2008-12-31'),
            ('2009 Recovery', '2009-01-01', '2009-04-30'),
            ('2011 Euro Debt', '2011-08-01', '2011-10-31'),
            ('2015 China Flash', '2015-08-01', '2015-09-30'),
            ('2018 December', '2018-12-01', '2018-12-31'),
            ('2020 COVID', '2020-02-15', '2020-04-30'),
            ('2022 Ukraine', '2022-02-15', '2022-04-30'),
        ]
        
        print(f"\nCrisis Period Analysis:")
        print(f"  {'Crisis':<25} {'Days':<8} {'Jumps':<8} {'Frequency':<12}")
        print(f"  {'-'*60}")
        
        for crisis_name, start, end in crisis_periods:
            crisis_stats = analyze_crisis_period(returns, threshold, crisis_name, start, end)
            if crisis_stats:
                print(f"  {crisis_stats['crisis']:<25} {crisis_stats['days']:<8} {crisis_stats['jumps']:<8} {crisis_stats['freq']:<12}")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    regime_325 = calculate_regime_stats(returns, 0.0325)
    regime_350 = calculate_regime_stats(returns, 0.035)
    
    print(f"\n3.25% threshold:")
    print(f"  ✓ JR2 segments: {regime_325['jr2_segments']} (target: 10-12 for meta-learning)")
    print(f"  ✓ Total jumps: {results[0.0325]['total_jumps']}")
    print(f"  ✓ Frequency: {results[0.0325]['jump_frequency']}")
    
    print(f"\n3.5% threshold:")
    print(f"  ✓ JR2 segments: {regime_350['jr2_segments']} (current)")
    print(f"  ✓ Total jumps: {results[0.035]['total_jumps']}")
    print(f"  ✓ Frequency: {results[0.035]['jump_frequency']}")
    
    # Decision
    if regime_325['jr2_segments'] >= 10 and regime_325['jr2_segments'] <= 12:
        print(f"\n🎯 VERDICT: 3.25% is OPTIMAL")
        print(f"   → Achieves target of 10-12 JR2 segments ({regime_325['jr2_segments']})")
        print(f"   → Better meta-learning diversity than 3.5% ({regime_350['jr2_segments']} segments)")
    elif regime_325['jr2_segments'] > 12:
        print(f"\n⚠️  VERDICT: 3.25% has TOO MANY JR2 segments ({regime_325['jr2_segments']})")
        print(f"   → Consider staying with 3.5% ({regime_350['jr2_segments']} segments)")
    else:
        print(f"\n💡 VERDICT: Both thresholds work")
        print(f"   → 3.25%: {regime_325['jr2_segments']} JR2 segments (more diversity)")
        print(f"   → 3.5%:  {regime_350['jr2_segments']} JR2 segments (less noise)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
