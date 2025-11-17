"""
Statistical Jump Detection for S&P 500 Daily Returns

Implements:
1. Lee-Mykland (2008) test adapted for daily data
2. Barndorff-Nielsen-Shephard (2006) bi-power variation test

Usage:
    python scripts/jm/detect_jumps.py --method lee_mykland --threshold 3.5
    python scripts/jm/detect_jumps.py --method bns --significance 0.01
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def lee_mykland_test(returns, window=20, threshold=4.6):
    """
    Lee-Mykland (2008) jump test adapted for daily data.
    
    Original uses high-frequency data, we adapt for daily:
    - Estimate local volatility with rolling window
    - Standardize returns by estimated volatility
    - Jumps = extreme standardized returns (>threshold std devs)
    
    Args:
        returns: pd.Series of daily returns
        window: Rolling window for volatility estimation (default 20 days)
        threshold: Number of std devs for jump detection (default 4.6σ as per Lee-Mykland)
        
    Returns:
        pd.DataFrame with columns: jump_flag, L_stat, threshold, jump_size
    """
    # Estimate local volatility (rolling std)
    rolling_vol = returns.rolling(window=window, min_periods=5).std()
    
    # Standardized returns (L-statistic)
    L_stat = returns / rolling_vol
    
    # Jump indicator (absolute L-stat exceeds threshold)
    jump_flag = (L_stat.abs() > threshold).astype(int)
    
    # Jump size (actual return on jump days)
    jump_size = returns.where(jump_flag == 1, 0)
    
    results = pd.DataFrame({
        'jump_flag': jump_flag,
        'L_stat': L_stat,
        'threshold': threshold,
        'jump_size': jump_size,
        'abs_L_stat': L_stat.abs()
    }, index=returns.index)
    
    return results


def bns_test(returns, window=20, significance=0.01):
    """
    Barndorff-Nielsen-Shephard (2006) bi-power variation test.
    
    Separates continuous volatility from jumps:
    - RV = realized variance (sum of squared returns)
    - BV = bi-power variation (robust to jumps)
    - Jump component = RV - BV
    
    Args:
        returns: pd.Series of daily returns
        window: Rolling window for RV/BV calculation
        significance: Significance level for jump test (default 0.01)
        
    Returns:
        pd.DataFrame with columns: jump_flag, Z_stat, RV, BV, jump_component
    """
    # Realized variance (sum of squared returns)
    RV = returns.rolling(window=window).apply(lambda x: (x**2).sum(), raw=True)
    
    # Bi-power variation: (π/2) * Σ|r_t||r_{t-1}|
    abs_returns = returns.abs()
    product_abs = abs_returns * abs_returns.shift(1)
    BV = (np.pi / 2) * product_abs.rolling(window=window).sum()
    
    # Jump component
    J = RV - BV
    
    # Test statistic (approximation for daily data)
    # Z ~ N(0,1) under null hypothesis of no jumps
    theta = (np.pi**2 / 4) + (np.pi - 5)  # Asymptotic variance constant
    Z_stat = J / np.sqrt(theta * BV)
    
    # Critical value from standard normal (e.g., 2.33 for 1% level)
    critical_value = -np.log(significance)  # Approximation
    
    # Jump indicator
    jump_flag = (Z_stat > critical_value).astype(int)
    
    results = pd.DataFrame({
        'jump_flag': jump_flag,
        'Z_stat': Z_stat,
        'RV': RV,
        'BV': BV,
        'jump_component': J,
        'critical_value': critical_value
    }, index=returns.index)
    
    return results


def simple_threshold_test(returns, threshold_pct=0.035):
    """
    Simple threshold test: Mark days with |return| > threshold as jumps.
    
    Args:
        returns: pd.Series of daily returns
        threshold_pct: Absolute return threshold (default 3.5%)
        
    Returns:
        pd.DataFrame with jump_flag and jump_size
    """
    abs_returns = returns.abs()
    jump_flag = (abs_returns > threshold_pct).astype(int)
    jump_size = returns.where(jump_flag == 1, 0)
    
    results = pd.DataFrame({
        'jump_flag': jump_flag,
        'jump_size': jump_size,
        'abs_return': abs_returns
    }, index=returns.index)
    
    return results


def visualize_jumps(df, jump_results, method_name, output_path):
    """Create visualization of detected jumps."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Panel 1: Price with jump markers
    ax1 = axes[0]
    ax1.plot(df.index, df['sp500_close'], label='S&P 500', alpha=0.7)
    jump_dates = jump_results[jump_results['jump_flag'] == 1].index
    ax1.scatter(jump_dates, df.loc[jump_dates, 'sp500_close'], 
                color='red', s=50, alpha=0.6, label=f'Jumps ({len(jump_dates)})', zorder=5)
    ax1.set_ylabel('S&P 500 Close')
    ax1.set_title(f'Jump Detection: {method_name}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Returns with jumps
    ax2 = axes[1]
    ax2.bar(df.index, df['sp500_return'], alpha=0.5, label='Returns', width=1)
    ax2.scatter(jump_dates, jump_results.loc[jump_dates, 'jump_size'],
                color='red', s=50, alpha=0.8, label='Jump Returns', zorder=5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Daily Return')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel 3: VIX with jumps
    ax3 = axes[2]
    ax3.plot(df.index, df['vix_close'], label='VIX', alpha=0.7, color='orange')
    ax3.scatter(jump_dates, df.loc[jump_dates, 'vix_close'],
                color='red', s=50, alpha=0.6, label='Jump Days VIX', zorder=5)
    ax3.axhline(y=40, color='red', linestyle='--', label='VIX 40 (HMM threshold)', alpha=0.5)
    ax3.axhline(y=30, color='orange', linestyle='--', label='VIX 30', alpha=0.5)
    ax3.set_ylabel('VIX')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved jump visualization to {output_path}")
    plt.close()


def analyze_crisis_periods(df, jump_results):
    """Analyze jump detection during known crisis periods."""
    crisis_periods = [
        ('2008 Financial Crisis', '2008-09-01', '2008-12-31'),
        ('2009 Continued', '2009-01-01', '2009-04-30'),
        ('2011 Euro Debt', '2011-08-01', '2011-10-31'),
        ('2015 China Flash', '2015-08-01', '2015-09-30'),
        ('2018 December', '2018-12-01', '2018-12-31'),
        ('2020 COVID', '2020-02-15', '2020-04-30'),
        ('2022 Ukraine', '2022-02-15', '2022-04-30'),
    ]
    
    print("\n" + "="*70)
    print("CRISIS PERIOD ANALYSIS")
    print("="*70)
    
    for crisis_name, start_date, end_date in crisis_periods:
        mask = (df.index >= start_date) & (df.index <= end_date)
        crisis_df = df[mask]
        crisis_jumps = jump_results[mask]
        
        if len(crisis_df) == 0:
            continue
            
        n_jumps = crisis_jumps['jump_flag'].sum()
        n_days = len(crisis_df)
        jump_freq = n_jumps / n_days if n_days > 0 else 0
        
        vix_max = crisis_df['vix_close'].max()
        vix_days_40 = (crisis_df['vix_close'] > 40).sum()
        vix_days_30 = (crisis_df['vix_close'] > 30).sum()
        
        avg_jump_size = crisis_jumps.loc[crisis_jumps['jump_flag'] == 1, 'jump_size'].abs().mean()
        negative_jumps = (crisis_jumps.loc[crisis_jumps['jump_flag'] == 1, 'jump_size'] < 0).sum()
        
        print(f"\n{crisis_name} ({start_date} to {end_date}):")
        print(f"  Total days: {n_days}")
        print(f"  Jumps detected: {n_jumps} ({jump_freq:.1%} frequency)")
        print(f"  Avg jump size: {avg_jump_size:.2%}" if not np.isnan(avg_jump_size) else "  Avg jump size: N/A")
        print(f"  Negative jumps: {negative_jumps}/{n_jumps}" if n_jumps > 0 else "  Negative jumps: 0/0")
        print(f"  Max VIX: {vix_max:.1f}")
        print(f"  Days VIX > 40: {vix_days_40} (HMM R2 criterion)")
        print(f"  Days VIX > 30: {vix_days_30}")
        
        # Highlight if JM detects but HMM misses
        if n_jumps > 0 and vix_days_40 == 0:
            print(f"  ⚠️  JM DETECTED {n_jumps} JUMPS BUT HMM MISSED (VIX < 40)")


def main():
    parser = argparse.ArgumentParser(description='Detect jumps in S&P 500 returns')
    parser.add_argument('--method', type=str, default='lee_mykland',
                       choices=['lee_mykland', 'bns', 'simple'],
                       help='Jump detection method')
    parser.add_argument('--threshold', type=float, default=4.6,
                       help='Threshold for Lee-Mykland test (std devs, default 4.6σ)')
    parser.add_argument('--significance', type=float, default=0.01,
                       help='Significance level for BNS test')
    parser.add_argument('--simple_threshold', type=float, default=0.035,
                       help='Return threshold for simple test (default 3.5%)')
    parser.add_argument('--window', type=int, default=20,
                       help='Rolling window for volatility estimation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path('data/processed/sp500_vix_merged_clean.csv')
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Total observations: {len(df)}")
    print(f"Columns: {list(df.columns[:10])}")
    
    # Calculate returns if not present
    if 'sp500_return' not in df.columns:
        if 'sp500_close' in df.columns:
            df['sp500_return'] = df['sp500_close'].pct_change()
        elif 'sp500_adj_close' in df.columns:
            df['sp500_return'] = df['sp500_adj_close'].pct_change()
    
    returns = df['sp500_return'].dropna()
    
    # Run jump detection
    print(f"\nRunning {args.method} jump detection...")
    
    if args.method == 'lee_mykland':
        jump_results = lee_mykland_test(returns, window=args.window, threshold=args.threshold)
        method_name = f"Lee-Mykland (threshold={args.threshold}σ)"
    elif args.method == 'bns':
        jump_results = bns_test(returns, window=args.window, significance=args.significance)
        method_name = f"BNS (α={args.significance})"
    else:  # simple
        jump_results = simple_threshold_test(returns, threshold_pct=args.simple_threshold)
        method_name = f"Simple (|r| > {args.simple_threshold:.1%})"
    
    # Merge with original data
    df = df.join(jump_results, how='left')
    
    # Summary statistics
    total_jumps = df['jump_flag'].sum()
    jump_freq = total_jumps / len(df)
    
    print(f"\n" + "="*70)
    print(f"JUMP DETECTION SUMMARY - {method_name}")
    print("="*70)
    print(f"Total jumps detected: {total_jumps}")
    print(f"Jump frequency: {jump_freq:.2%} ({total_jumps}/{len(df)} days)")
    print(f"Avg time between jumps: {1/jump_freq:.1f} days" if jump_freq > 0 else "No jumps")
    
    # Jump direction analysis
    positive_jumps = (df.loc[df['jump_flag'] == 1, 'jump_size'] > 0).sum()
    negative_jumps = (df.loc[df['jump_flag'] == 1, 'jump_size'] < 0).sum()
    print(f"\nJump direction:")
    print(f"  Positive (rally): {positive_jumps} ({positive_jumps/total_jumps:.1%})")
    print(f"  Negative (crash): {negative_jumps} ({negative_jumps/total_jumps:.1%})")
    
    # Largest jumps
    print(f"\nLargest jumps detected:")
    largest = df[df['jump_flag'] == 1].nlargest(10, 'jump_size')[['jump_size', 'vix_close']]
    for date, row in largest.iterrows():
        print(f"  {date.date()}: {row['jump_size']:+.2%} (VIX: {row['vix_close']:.1f})")
    
    print(f"\nLargest negative jumps:")
    smallest = df[df['jump_flag'] == 1].nsmallest(10, 'jump_size')[['jump_size', 'vix_close']]
    for date, row in smallest.iterrows():
        print(f"  {date.date()}: {row['jump_size']:+.2%} (VIX: {row['vix_close']:.1f})")
    
    # Crisis period analysis
    analyze_crisis_periods(df, df)
    
    # Save results
    output_path = Path(f'data/jm/jump_indicators_{args.method}.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\n✅ Saved jump indicators to {output_path}")
    
    # Visualization
    if args.visualize:
        viz_path = Path(f'docs/jm/figures/jumps_{args.method}.png')
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_jumps(df, df, method_name, viz_path)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review crisis period analysis above")
    print("2. Check if JM detects 2022 Ukraine (HMM missed it)")
    print("3. Run: python scripts/jm/classify_jump_regimes.py")
    print("4. Compare JM regimes vs HMM regimes (VIX threshold)")


if __name__ == '__main__':
    main()
