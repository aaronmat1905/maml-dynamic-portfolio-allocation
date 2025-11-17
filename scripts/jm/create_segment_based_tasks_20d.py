"""
Create segment-based tasks with LOWER THRESHOLD (20 days instead of 30 days).

This captures short but intense crises that were missed with 30-day minimum:
- 1987 Black Monday (5-10 days)
- 2010 Flash Crash (3-7 days)
- 2015 China Crash (15-20 days)

Expected result: 5 JR2 tasks → 8-12 JR2 tasks

Author: PreethamVJ
Date: November 2025
Version: 20-day threshold experiment
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
JM_DIR = DATA_DIR / "jm"

# Input files
SEGMENTS_FILE = JM_DIR / "jump_regime_segments.csv"
REGIME_LABELS_FILE = JM_DIR / "jump_regime_labels.csv"
JUMP_INDICATORS_FILE = JM_DIR / "jump_indicators_simple.csv"

# Output files (separate from original)
OUTPUT_DIR = JM_DIR / "20d_threshold"
OUTPUT_DIR.mkdir(exist_ok=True)
SEGMENT_TASKS_FILE = OUTPUT_DIR / "tasks_metadata_20d.json"
SEGMENT_SPLIT_FILE = OUTPUT_DIR / "task_split_indices_20d.json"

# Parameters - MODIFIED FROM ORIGINAL
MIN_TASK_LENGTH = 20      # ← CHANGED from 30 to 20
TARGET_TASK_LENGTH = 30   # ← CHANGED from 40 to 30
MIN_REGIME_PURITY = 0.55  # ← CHANGED from implicit 1.0 to 0.55
TRAIN_RATIO = 0.52
VAL_RATIO = 0.13


def load_data():
    """Load all necessary data files."""
    print("Loading data files...")
    
    segments_df = pd.read_csv(SEGMENTS_FILE, parse_dates=['start_date', 'end_date'])
    regime_df = pd.read_csv(REGIME_LABELS_FILE, parse_dates=['Date'], index_col='Date')
    jump_df = pd.read_csv(JUMP_INDICATORS_FILE, parse_dates=['Date'], index_col='Date')
    
    print(f"✓ Loaded {len(segments_df)} regime segments")
    print(f"✓ Loaded {len(regime_df)} daily regime labels")
    print(f"✓ Loaded {len(jump_df)} daily jump indicators")
    
    return segments_df, regime_df, jump_df


def calculate_task_features(regime_df, jump_df, start_date, end_date):
    """Calculate jump features for a task window."""
    # Get data for this window
    window_regime = regime_df.loc[start_date:end_date]
    window_jump = jump_df.loc[start_date:end_date]
    
    # Jump frequency
    n_jumps = window_jump['jump_flag'].sum()
    n_days = len(window_jump)
    jump_frequency = n_jumps / n_days if n_days > 0 else 0
    
    # Average jump size
    jump_returns = window_jump[window_jump['jump_flag'] == 1]['sp500_adj_close_log_return']
    avg_jump_size = jump_returns.abs().mean() if len(jump_returns) > 0 else 0
    
    # Jump clustering
    if n_jumps > 1:
        jump_indices = np.where(window_jump['jump_flag'] == 1)[0]
        gaps = np.diff(jump_indices)
        jump_clustering = gaps.std() / gaps.mean() if gaps.mean() > 0 else 0
    else:
        jump_clustering = np.nan
    
    # Negative jump ratio
    if n_jumps > 0:
        n_negative = (jump_returns < 0).sum()
        negative_jump_ratio = n_negative / n_jumps
    else:
        negative_jump_ratio = 0.5
    
    # VIX stats
    max_vix = window_jump['vix_close'].max() if 'vix_close' in window_jump.columns else None
    avg_vix = window_jump['vix_close'].mean() if 'vix_close' in window_jump.columns else None
    
    return {
        'jump_frequency': float(jump_frequency),
        'avg_jump_size': float(avg_jump_size),
        'jump_clustering': float(jump_clustering) if not np.isnan(jump_clustering) else None,
        'negative_jump_ratio': float(negative_jump_ratio),
        'max_vix': float(max_vix) if max_vix is not None else None,
        'avg_vix': float(avg_vix) if avg_vix is not None else None
    }


def create_segment_based_tasks(segments_df, regime_df, jump_df):
    """Create tasks with LOWER 20-day threshold."""
    print("\n" + "=" * 80)
    print("CREATING SEGMENT-BASED TASKS (20-DAY THRESHOLD)")
    print("=" * 80)
    print(f"Min Task Length: {MIN_TASK_LENGTH} days (was 30)")
    print(f"Target Length: {TARGET_TASK_LENGTH} days (was 40)")
    print(f"Min Purity: {MIN_REGIME_PURITY:.0%} (was 100%)")
    
    tasks = []
    task_id = 0
    
    # Process ALL segments
    for regime_type in [0, 1, 2]:
        regime_segments = segments_df[segments_df['regime'] == regime_type].copy()
        
        print(f"\nProcessing Regime {regime_type} ({len(regime_segments)} segments):")
        
        for idx, segment in regime_segments.iterrows():
            start_date = pd.to_datetime(segment['start_date'])
            end_date = pd.to_datetime(segment['end_date'])
            duration = int(segment['length'])
            
            # CHANGED: Now accepts 20+ days (was 30+)
            if duration < MIN_TASK_LENGTH:
                print(f"  ✗ Skipping {start_date.date()} - {end_date.date()} ({duration}d) - too short (<{MIN_TASK_LENGTH}d)")
                continue
            
            # Option 1: Use segment as-is
            if duration >= TARGET_TASK_LENGTH:
                task_start = start_date
                task_end = end_date
                task_length = duration
                purity = 1.0
                source = 'segment_exact'
                
            # Option 2: Pad to target length
            else:
                needed_padding = TARGET_TASK_LENGTH - duration
                expand_before = needed_padding // 2
                expand_after = needed_padding - expand_before
                
                task_start = start_date - pd.Timedelta(days=expand_before)
                task_end = end_date + pd.Timedelta(days=expand_after)
                task_length = TARGET_TASK_LENGTH
                purity = duration / TARGET_TASK_LENGTH
                source = 'segment_padded'
            
            # CHANGED: Check purity threshold
            if purity < MIN_REGIME_PURITY:
                print(f"  ✗ Skipping {start_date.date()} - low purity ({purity:.1%} < {MIN_REGIME_PURITY:.0%})")
                continue
            
            # Ensure dates within bounds
            if task_start < regime_df.index[0]:
                task_start = regime_df.index[0]
            if task_end > regime_df.index[-1]:
                task_end = regime_df.index[-1]
            
            # Get indices
            try:
                start_idx = regime_df.index.get_loc(task_start)
            except KeyError:
                start_idx = regime_df.index.get_indexer([task_start], method='nearest')[0]
            
            try:
                end_idx = regime_df.index.get_loc(task_end)
            except KeyError:
                end_idx = regime_df.index.get_indexer([task_end], method='nearest')[0]
            
            actual_length = end_idx - start_idx
            
            # Update to actual dates
            task_start = regime_df.index[start_idx]
            task_end = regime_df.index[end_idx]
            
            # Calculate features
            features = calculate_task_features(regime_df, jump_df, task_start, task_end)
            
            # Create task
            task = {
                'task_id': task_id,
                'regime': int(regime_type),
                'regime_purity': float(purity),
                'start_date': task_start.strftime('%Y-%m-%d'),
                'end_date': task_end.strftime('%Y-%m-%d'),
                'length': int(actual_length),
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'source': source,
                'segment_duration': int(duration),
                **features
            }
            
            tasks.append(task)
            
            # Print JR2 details
            if regime_type == 2:
                print(f"  ✓ Task {task_id}: {start_date.date()} - {end_date.date()} "
                      f"({duration}d segment → {actual_length}d task, purity={purity:.2f})")
            
            task_id += 1
    
    return tasks


def split_tasks(tasks):
    """Split tasks chronologically."""
    print("\n" + "=" * 80)
    print("SPLITTING TASKS INTO TRAIN/VAL/TEST")
    print("=" * 80)
    
    # Sort by start date
    tasks_sorted = sorted(tasks, key=lambda x: x['start_date'])
    n_tasks = len(tasks_sorted)
    
    # Calculate split points
    train_end = int(n_tasks * TRAIN_RATIO)
    val_end = train_end + int(n_tasks * VAL_RATIO)
    
    # Create indices
    train_indices = [t['task_id'] for t in tasks_sorted[:train_end]]
    val_indices = [t['task_id'] for t in tasks_sorted[train_end:val_end]]
    test_indices = [t['task_id'] for t in tasks_sorted[val_end:]]
    
    split = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'experiment': '20d_threshold',
        'min_task_length': MIN_TASK_LENGTH,
        'target_task_length': TARGET_TASK_LENGTH
    }
    
    # Print statistics
    print(f"\nTotal Tasks: {n_tasks}")
    print(f"Train: {len(train_indices)} tasks ({len(train_indices)/n_tasks*100:.1f}%)")
    print(f"Val:   {len(val_indices)} tasks ({len(val_indices)/n_tasks*100:.1f}%)")
    print(f"Test:  {len(test_indices)} tasks ({len(test_indices)/n_tasks*100:.1f}%)")
    
    # Regime distribution
    for split_name, indices in split.items():
        if split_name in ['experiment', 'min_task_length', 'target_task_length']:
            continue
            
        split_tasks = [t for t in tasks if t['task_id'] in indices]
        regime_counts = {0: 0, 1: 0, 2: 0}
        for t in split_tasks:
            regime_counts[t['regime']] += 1
        
        print(f"\n{split_name.upper()} regime distribution:")
        print(f"  JR0: {regime_counts[0]} tasks")
        print(f"  JR1: {regime_counts[1]} tasks")
        print(f"  JR2: {regime_counts[2]} tasks ⭐")
    
    return split


def print_jr2_analysis(tasks, original_jr2_count=5):
    """Print detailed JR2 analysis with comparison."""
    jr2_tasks = [t for t in tasks if t['regime'] == 2]
    
    print("\n" + "=" * 80)
    print(f"JR2 CRISIS TASKS ANALYSIS")
    print("=" * 80)
    print(f"Original (30d threshold): {original_jr2_count} tasks")
    print(f"New (20d threshold):      {len(jr2_tasks)} tasks")
    print(f"Improvement:              +{len(jr2_tasks) - original_jr2_count} tasks ({(len(jr2_tasks)/original_jr2_count - 1)*100:+.0f}%)")
    
    crisis_mapping = {
        '1987-10': 'Black Monday 1987 ⭐ NEW!',
        '1998-08': 'LTCM Collapse',
        '2002-07': 'WorldCom/Enron Crisis',
        '2002-10': 'Dot-com Bottom',
        '2008-09': '2008 Financial Crisis (Lehman)',
        '2009-02': '2008 Aftermath',
        '2010-05': 'Flash Crash 2010 ⭐ NEW!',
        '2011-08': 'US Debt Downgrade',
        '2015-08': 'China Stock Crash ⭐ NEW!',
        '2018-12': 'Dec 2018 Selloff ⭐ NEW!',
        '2020-03': 'COVID-19 Crash',
    }
    
    print("\nDetailed Crisis Tasks:")
    for task in sorted(jr2_tasks, key=lambda x: x['start_date']):
        start = task['start_date']
        year_month = start[:7]
        crisis_name = crisis_mapping.get(year_month, 'Unknown Crisis')
        
        print(f"\nTask {task['task_id']}: {crisis_name}")
        print(f"  Date Range: {task['start_date']} to {task['end_date']}")
        print(f"  Length: {task['length']} days (segment: {task['segment_duration']}d)")
        print(f"  Purity: {task['regime_purity']:.2%}")
        print(f"  Jump Freq: {task['jump_frequency']:.1%}")
        print(f"  Avg Jump Size: {task['avg_jump_size']:.2%}")


def main():
    """Main execution."""
    print("=" * 80)
    print("SEGMENT-BASED TASK GENERATION - 20-DAY THRESHOLD")
    print("=" * 80)
    print(f"\n🎯 EXPERIMENT: Lower threshold to capture more short crises")
    print(f"\nParameters:")
    print(f"  Min Task Length: {MIN_TASK_LENGTH} days (was 30)")
    print(f"  Target Length: {TARGET_TASK_LENGTH} days (was 40)")
    print(f"  Min Purity: {MIN_REGIME_PURITY:.0%} (was 100%)")
    print(f"  Train/Val/Test: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{1-TRAIN_RATIO-VAL_RATIO:.0%}")
    
    # Load data
    segments_df, regime_df, jump_df = load_data()
    
    # Create tasks
    tasks = create_segment_based_tasks(segments_df, regime_df, jump_df)
    
    # Split tasks
    split = split_tasks(tasks)
    
    # Print JR2 analysis
    print_jr2_analysis(tasks)
    
    # Save outputs
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)
    
    with open(SEGMENT_TASKS_FILE, 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"✓ Saved {len(tasks)} tasks to {SEGMENT_TASKS_FILE}")
    
    with open(SEGMENT_SPLIT_FILE, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"✓ Saved split indices to {SEGMENT_SPLIT_FILE}")
    
    # Final summary
    jr2_count = sum(1 for t in tasks if t['regime'] == 2)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Tasks: {len(tasks)}")
    print(f"JR0 Tasks: {sum(1 for t in tasks if t['regime'] == 0)}")
    print(f"JR1 Tasks: {sum(1 for t in tasks if t['regime'] == 1)}")
    print(f"JR2 Tasks: {jr2_count} (was 5 with 30d threshold)")
    
    if jr2_count >= 8:
        print(f"\n✅ SUCCESS: {jr2_count} JR2 tasks (60% increase!)")
        print(f"✅ Ready to retrain MAML with more crisis diversity")
    elif jr2_count > 5:
        print(f"\n✅ MODERATE SUCCESS: {jr2_count} JR2 tasks (+{jr2_count-5})")
        print(f"✅ Some improvement, worth retraining")
    else:
        print(f"\n⚠️  LIMITED IMPROVEMENT: Only {jr2_count} JR2 tasks")
        print(f"⚠️  May need to combine with other data sources")


if __name__ == "__main__":
    main()
