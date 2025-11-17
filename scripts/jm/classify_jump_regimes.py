"""
Jump Regime Classification for MAML Tasks

Creates regime labels based on jump intensity (frequency + size) instead of VIX thresholds.

Regimes:
- JR0 (Calm): <1 jump per month (λ < 0.05 daily)
- JR1 (Moderate): 1-3 jumps per month (0.05 ≤ λ < 0.15)
- JR2 (Extreme): >3 jumps per month (λ ≥ 0.15) + large avg jump size

Usage:
    python scripts/jm/classify_jump_regimes.py --input data/jm/jump_indicators_simple.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_jump_intensity(df, window=20):
    """
    Calculate rolling jump intensity (frequency) and average jump size.
    
    Args:
        df: DataFrame with jump_flag column
        window: Rolling window for intensity calculation (default 20 days ~ 1 month)
        
    Returns:
        DataFrame with jump_intensity and avg_jump_size columns
    """
    # Jump frequency (proportion of days with jumps in rolling window)
    df['jump_intensity'] = df['jump_flag'].rolling(window=window, min_periods=1).mean()
    
    # Average absolute jump size in rolling window
    df['avg_jump_size'] = df['jump_size'].abs().rolling(window=window, min_periods=1).mean()
    
    # Jump clustering (autocorrelation of jump indicator)
    # High clustering = jumps tend to occur in bursts
    df['jump_clustering'] = df['jump_flag'].rolling(window=window, min_periods=2).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
    )
    
    # Negative jump ratio (proportion of jumps that are negative)
    def neg_jump_ratio(series):
        jumps = series[series != 0]
        if len(jumps) == 0:
            return 0.5  # Neutral if no jumps
        return (jumps < 0).sum() / len(jumps)
    
    df['negative_jump_ratio'] = df['jump_size'].rolling(window=window, min_periods=1).apply(
        neg_jump_ratio, raw=False
    )
    
    return df


def classify_regimes(df, intensity_thresholds=(0.05, 0.15), size_threshold=0.03):
    """
    Classify each day into jump regime based on intensity and size.
    
    Args:
        df: DataFrame with jump_intensity and avg_jump_size columns
        intensity_thresholds: Tuple of (JR0_max, JR1_max) for jump frequency
        size_threshold: Minimum avg jump size for JR2 classification (default 3%)
        
    Returns:
        DataFrame with regime_label column
    """
    intensity_low, intensity_high = intensity_thresholds
    
    # Initialize regime labels
    regime_labels = pd.Series(0, index=df.index)  # Default JR0
    
    # JR1: Moderate jump intensity
    moderate_mask = (df['jump_intensity'] >= intensity_low) & (df['jump_intensity'] < intensity_high)
    regime_labels[moderate_mask] = 1
    
    # JR2: Extreme jump intensity OR high intensity + large jump size
    extreme_mask = (df['jump_intensity'] >= intensity_high) | \
                   ((df['jump_intensity'] >= intensity_low) & (df['avg_jump_size'] >= size_threshold))
    regime_labels[extreme_mask] = 2
    
    df['regime_label'] = regime_labels.astype(int)
    
    return df


def identify_regime_segments(df, min_segment_length=5):
    """
    Identify continuous segments of each regime (for task creation).
    
    Args:
        df: DataFrame with regime_label column
        min_segment_length: Minimum consecutive days to count as a segment
        
    Returns:
        List of dicts with segment info: {regime, start_date, end_date, length}
    """
    segments = []
    current_regime = None
    segment_start = None
    
    for date, row in df.iterrows():
        regime = row['regime_label']
        
        if regime != current_regime:
            # Save previous segment if long enough
            if current_regime is not None and segment_start is not None:
                segment_length = (date - segment_start).days
                if segment_length >= min_segment_length:
                    segments.append({
                        'regime': current_regime,
                        'start_date': segment_start,
                        'end_date': date,
                        'length': segment_length
                    })
            
            # Start new segment
            current_regime = regime
            segment_start = date
    
    # Add final segment
    if current_regime is not None and segment_start is not None:
        segment_length = (df.index[-1] - segment_start).days
        if segment_length >= min_segment_length:
            segments.append({
                'regime': current_regime,
                'start_date': segment_start,
                'end_date': df.index[-1],
                'length': segment_length
            })
    
    return segments


def visualize_regimes(df, output_path):
    """Create visualization of jump regimes over time."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    # Panel 1: S&P 500 with regime colors
    ax1 = axes[0]
    for regime in [0, 1, 2]:
        mask = df['regime_label'] == regime
        regime_names = {0: 'JR0 (Calm)', 1: 'JR1 (Moderate)', 2: 'JR2 (Extreme)'}
        colors = {0: 'green', 1: 'orange', 2: 'red'}
        ax1.scatter(df[mask].index, df[mask]['sp500_close'], 
                   c=colors[regime], s=1, alpha=0.3, label=regime_names[regime])
    ax1.set_ylabel('S&P 500 Close')
    ax1.set_title('Jump Regime Classification Over Time')
    ax1.legend(markerscale=10)
    ax1.grid(alpha=0.3)
    
    # Panel 2: Jump intensity with regime thresholds
    ax2 = axes[1]
    ax2.fill_between(df.index, 0, df['jump_intensity'], alpha=0.3, color='blue')
    ax2.axhline(y=0.05, color='orange', linestyle='--', label='JR0/JR1 threshold', alpha=0.7)
    ax2.axhline(y=0.15, color='red', linestyle='--', label='JR1/JR2 threshold', alpha=0.7)
    ax2.set_ylabel('Jump Intensity (20-day)')
    ax2.set_ylim(0, max(0.6, df['jump_intensity'].max() * 1.1))
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel 3: VIX with HMM threshold comparison
    ax3 = axes[2]
    ax3.plot(df.index, df['vix_close'], alpha=0.7, color='purple', linewidth=0.5)
    ax3.axhline(y=40, color='red', linestyle='--', label='HMM threshold (VIX 40)', linewidth=2)
    ax3.axhline(y=30, color='orange', linestyle='--', label='Alternative (VIX 30)', alpha=0.5)
    ax3.set_ylabel('VIX Close')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Panel 4: Regime timeline (bar chart)
    ax4 = axes[3]
    regime_colors = {0: 'green', 1: 'orange', 2: 'red'}
    for idx, (date, row) in enumerate(df.iterrows()):
        if idx % 10 == 0:  # Subsample for performance
            ax4.axvline(x=date, color=regime_colors[row['regime_label']], alpha=0.3, linewidth=1)
    ax4.set_ylabel('Regime')
    ax4.set_xlabel('Date')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['JR0', 'JR1', 'JR2'])
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved regime visualization to {output_path}")
    plt.close()


def compare_with_hmm(df, hmm_threshold=40):
    """Compare jump regimes with HMM VIX threshold approach."""
    # Create HMM-style regime (simplified: VIX > threshold = R2)
    df['hmm_regime'] = (df['vix_close'] > hmm_threshold).astype(int) * 2  # 0 or 2
    
    print("\n" + "="*70)
    print("JUMP REGIMES VS HMM COMPARISON")
    print("="*70)
    
    # Overall distribution
    print("\nRegime distribution:")
    print("\nJump Model (JM):")
    jr_counts = df['regime_label'].value_counts().sort_index()
    for regime, count in jr_counts.items():
        pct = count / len(df) * 100
        regime_name = {0: 'JR0 (Calm)', 1: 'JR1 (Moderate)', 2: 'JR2 (Extreme)'}[regime]
        print(f"  {regime_name}: {count} days ({pct:.1f}%)")
    
    print("\nHMM (VIX > 40):")
    hmm_counts = df['hmm_regime'].value_counts().sort_index()
    for regime, count in hmm_counts.items():
        pct = count / len(df) * 100
        regime_name = {0: 'R0/R1 (Normal)', 2: 'R2 (Crisis)'}[regime]
        print(f"  {regime_name}: {count} days ({pct:.1f}%)")
    
    # Crisis periods comparison
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
    print("CRISIS DETECTION COMPARISON")
    print("="*70)
    
    for crisis_name, start, end in crisis_periods:
        mask = (df.index >= start) & (df.index <= end)
        crisis_df = df[mask]
        
        if len(crisis_df) == 0:
            continue
        
        jm_jr2_days = (crisis_df['regime_label'] == 2).sum()
        hmm_r2_days = (crisis_df['hmm_regime'] == 2).sum()
        total_days = len(crisis_df)
        
        print(f"\n{crisis_name}:")
        print(f"  JM detected: {jm_jr2_days}/{total_days} days as JR2 ({jm_jr2_days/total_days*100:.1f}%)")
        print(f"  HMM detected: {hmm_r2_days}/{total_days} days as R2 ({hmm_r2_days/total_days*100:.1f}%)")
        
        if jm_jr2_days > 0 and hmm_r2_days == 0:
            print(f"  ✅ JM CAPTURES THIS CRISIS, HMM MISSES IT!")
        elif jm_jr2_days == 0 and hmm_r2_days > 0:
            print(f"  ⚠️  HMM captures, JM misses")
        elif jm_jr2_days > 0 and hmm_r2_days > 0:
            print(f"  ✓ Both detect this crisis")


def main():
    parser = argparse.ArgumentParser(description='Classify jump regimes for MAML')
    parser.add_argument('--input', type=str, default='data/jm/jump_indicators_simple.csv',
                       help='Input CSV with jump indicators')
    parser.add_argument('--window', type=int, default=20,
                       help='Rolling window for intensity calculation (default 20 days)')
    parser.add_argument('--intensity_low', type=float, default=0.05,
                       help='JR0/JR1 threshold (default 0.05 = 1 jump/month)')
    parser.add_argument('--intensity_high', type=float, default=0.15,
                       help='JR1/JR2 threshold (default 0.15 = 3 jumps/month)')
    parser.add_argument('--size_threshold', type=float, default=0.03,
                       help='Avg jump size threshold for JR2 (default 3%)')
    parser.add_argument('--min_segment_length', type=int, default=10,
                       help='Minimum segment length for task creation (default 10 days)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Load jump indicators
    input_path = Path(args.input)
    print(f"Loading jump indicators from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Total observations: {len(df)}")
    
    # Calculate jump intensity metrics
    print(f"\nCalculating jump intensity (window={args.window} days)...")
    df = calculate_jump_intensity(df, window=args.window)
    
    # Classify regimes
    print(f"\nClassifying regimes (thresholds: {args.intensity_low}, {args.intensity_high})...")
    df = classify_regimes(df, 
                         intensity_thresholds=(args.intensity_low, args.intensity_high),
                         size_threshold=args.size_threshold)
    
    # Summary statistics
    print("\n" + "="*70)
    print("REGIME CLASSIFICATION SUMMARY")
    print("="*70)
    
    regime_counts = df['regime_label'].value_counts().sort_index()
    regime_names = {0: 'JR0 (Calm)', 1: 'JR1 (Moderate)', 2: 'JR2 (Extreme)'}
    
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        avg_intensity = df[df['regime_label'] == regime]['jump_intensity'].mean()
        avg_size = df[df['regime_label'] == regime]['avg_jump_size'].mean()
        
        print(f"\n{regime_names[regime]}:")
        print(f"  Total days: {count} ({pct:.1f}%)")
        print(f"  Avg jump intensity: {avg_intensity:.2%}")
        print(f"  Avg jump size: {avg_size:.2%}")
    
    # Identify regime segments
    print(f"\nIdentifying regime segments (min length={args.min_segment_length} days)...")
    segments = identify_regime_segments(df, min_segment_length=args.min_segment_length)
    
    print(f"\nTotal segments found: {len(segments)}")
    for regime in [0, 1, 2]:
        regime_segs = [s for s in segments if s['regime'] == regime]
        avg_length = np.mean([s['length'] for s in regime_segs]) if regime_segs else 0
        print(f"  {regime_names[regime]}: {len(regime_segs)} segments (avg length: {avg_length:.1f} days)")
    
    # Show JR2 segments specifically (for MAML crisis tasks)
    jr2_segments = [s for s in segments if s['regime'] == 2]
    print(f"\n{regime_names[2]} segments (crisis periods for MAML):")
    for i, seg in enumerate(jr2_segments[:20], 1):  # Show first 20
        print(f"  {i}. {seg['start_date'].date()} to {seg['end_date'].date()} ({seg['length']} days)")
    
    if len(jr2_segments) > 20:
        print(f"  ... and {len(jr2_segments) - 20} more")
    
    # Compare with HMM
    compare_with_hmm(df, hmm_threshold=40)
    
    # Save regime labels
    output_path = Path('data/jm/jump_regime_labels.csv')
    df.to_csv(output_path)
    print(f"\n✅ Saved jump regime labels to {output_path}")
    
    # Save segments for task generation
    segments_df = pd.DataFrame(segments)
    segments_path = Path('data/jm/jump_regime_segments.csv')
    segments_df.to_csv(segments_path, index=False)
    print(f"✅ Saved regime segments to {segments_path}")
    
    # Visualization
    if args.visualize:
        viz_path = Path('docs/jm/figures/jump_regimes.png')
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_regimes(df, viz_path)
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    jr2_count = (df['regime_label'] == 2).sum()
    hmm_r2_count = (df['vix_close'] > 40).sum()
    print(f"JM JR2 days: {jr2_count} ({jr2_count/len(df)*100:.1f}%)")
    print(f"HMM R2 days: {hmm_r2_count} ({hmm_r2_count/len(df)*100:.1f}%)")
    print(f"JM detects {len(jr2_segments)} crisis segments vs HMM's 3 crisis periods")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review JR2 segments above - these become MAML crisis tasks")
    print("2. Check crisis comparison - does JM capture Ukraine 2022?")
    print("3. Run: python scripts/jm/generate_jump_tasks.py")
    print("4. Compare JM-MAML vs HMM-MAML performance")


if __name__ == '__main__':
    main()
