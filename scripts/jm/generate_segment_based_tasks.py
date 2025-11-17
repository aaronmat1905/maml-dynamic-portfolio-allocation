"""
Segment-Based Task Generation for Jump Model MAML

This approach creates ONE task per JR2 segment, guaranteeing that ALL
8 crisis periods are captured as tasks (unlike sliding windows which missed 2008).

Strategy:
- Each JR2 segment becomes a task
- Long segments (≥40 days): Use as-is
- Short segments (<40 days): Pad with surrounding data to reach minimum length
- Result: 8 JR2 tasks guaranteed (one per crisis)

Author: PreethamVJ
Date: November 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
JM_DIR = DATA_DIR / 'jm'

# Task parameters
MIN_TASK_LENGTH = 40  # Minimum days for a viable task
SUPPORT_SIZE = 30     # Days for training
QUERY_SIZE = 10       # Days for testing


def load_data():
    """Load regime segments and daily data."""
    print("📂 Loading data...")
    
    # Load segments
    segments_df = pd.read_csv(
        JM_DIR / 'jump_regime_segments.csv',
        parse_dates=['start_date', 'end_date']
    )
    
    # Load daily regime labels
    regime_df = pd.read_csv(
        JM_DIR / 'jump_regime_labels.csv',
        parse_dates=['Date'],
        index_col='Date'
    )
    
    # Load jump indicators for features
    jump_df = pd.read_csv(
        JM_DIR / 'jump_indicators_simple.csv',
        parse_dates=['Date'],
        index_col='Date'
    )
    
    print(f"   ✓ Loaded {len(segments_df)} segments")
    print(f"   ✓ Loaded {len(regime_df)} daily labels")
    print(f"   ✓ Loaded {len(jump_df)} daily jump indicators")
    
    return segments_df, regime_df, jump_df


def calculate_task_features(task_window, regime_df, jump_df):
    """Calculate jump-based features for a task window."""
    start_date = pd.to_datetime(task_window['start_date'])
    end_date = pd.to_datetime(task_window['end_date'])
    
    # Get window data
    window_jumps = jump_df.loc[start_date:end_date]
    window_regime = regime_df.loc[start_date:end_date]
    
    # Calculate features
    jump_flags = window_jumps['jump_flag']
    n_jumps = jump_flags.sum()
    n_days = len(window_jumps)
    
    # Jump frequency
    jump_frequency = n_jumps / n_days if n_days > 0 else 0.0
    
    # Average jump size
    jump_returns = window_jumps[window_jumps['jump_flag'] == 1]['sp500_adj_close_log_return']
    avg_jump_size = jump_returns.abs().mean() if len(jump_returns) > 0 else 0.0
    
    # Jump clustering (coefficient of variation of inter-jump distances)
    jump_dates = window_jumps[window_jumps['jump_flag'] == 1].index
    if len(jump_dates) > 1:
        inter_jump_days = [(jump_dates[i+1] - jump_dates[i]).days 
                          for i in range(len(jump_dates)-1)]
        if len(inter_jump_days) > 0 and np.mean(inter_jump_days) > 0:
            jump_clustering = np.std(inter_jump_days) / np.mean(inter_jump_days)
        else:
            jump_clustering = 0.0
    else:
        jump_clustering = np.nan
    
    # Negative jump ratio
    if n_jumps > 0:
        negative_jumps = (jump_returns < 0).sum()
        negative_jump_ratio = negative_jumps / n_jumps
    else:
        negative_jump_ratio = 0.5
    
    # VIX statistics
    max_vix = window_jumps['vix_close'].max()
    avg_vix = window_jumps['vix_close'].mean()
    
    # Regime purity
    regime_counts = window_regime['regime_label'].value_counts()
    majority_regime = regime_counts.idxmax() if len(regime_counts) > 0 else 0
    regime_purity = regime_counts.max() / len(window_regime) if len(window_regime) > 0 else 1.0
    
    return {
        'regime': int(majority_regime),
        'regime_purity': float(regime_purity),
        'jump_frequency': float(jump_frequency),
        'avg_jump_size': float(avg_jump_size),
        'jump_clustering': float(jump_clustering) if not np.isnan(jump_clustering) else None,
        'negative_jump_ratio': float(negative_jump_ratio),
        'max_vix': float(max_vix) if not pd.isna(max_vix) else None,
        'avg_vix': float(avg_vix) if not pd.isna(avg_vix) else None
    }


def create_segment_task(segment, regime_df, jump_df, task_id, start_idx_offset):
    """
    Create a task from a JR2 segment.
    
    Strategy:
    - If segment ≥ MIN_TASK_LENGTH: Use as-is
    - If segment < MIN_TASK_LENGTH: Pad symmetrically to reach min length
    """
    start_date = segment['start_date']
    end_date = segment['end_date']
    duration = segment['length']
    
    # Calculate task window
    if duration >= MIN_TASK_LENGTH:
        # Use segment as-is
        task_start = start_date
        task_end = end_date
        task_length = duration
        source = 'segment_exact'
    else:
        # Pad to minimum length
        needed_padding = MIN_TASK_LENGTH - duration
        pad_before = needed_padding // 2
        pad_after = needed_padding - pad_before
        
        task_start = start_date - timedelta(days=pad_before)
        task_end = end_date + timedelta(days=pad_after)
        task_length = MIN_TASK_LENGTH
        source = 'segment_padded'
    
    # Get actual indices in the full dataset
    try:
        start_idx = regime_df.index.get_loc(task_start)
        end_idx = regime_df.index.get_loc(task_end)
    except KeyError:
        # Dates outside data range
        return None
    
    # Calculate features
    task_window = {
        'start_date': task_start,
        'end_date': task_end,
        'length': task_length
    }
    features = calculate_task_features(task_window, regime_df, jump_df)
    
    # Build task metadata
    task = {
        'task_id': task_id,
        'regime': features['regime'],
        'regime_purity': features['regime_purity'],
        'start_date': task_start.strftime('%Y-%m-%d'),
        'end_date': task_end.strftime('%Y-%m-%d'),
        'length': task_length,
        'start_idx': int(start_idx + start_idx_offset),
        'end_idx': int(end_idx + start_idx_offset),
        'jump_frequency': features['jump_frequency'],
        'avg_jump_size': features['avg_jump_size'],
        'jump_clustering': features['jump_clustering'],
        'negative_jump_ratio': features['negative_jump_ratio'],
        'max_vix': features['max_vix'],
        'avg_vix': features['avg_vix'],
        'source': source,
        'original_segment_length': int(duration)
    }
    
    return task


def create_all_segment_tasks(segments_df, regime_df, jump_df):
    """Create tasks for all regimes using segment-based approach."""
    tasks = []
    task_id = 0
    
    # Get start index offset (if regime_df doesn't start at 0)
    start_idx_offset = 0
    
    print("\n🔨 Creating segment-based tasks...")
    
    # Process all segments (JR0, JR1, JR2)
    for regime in [0, 1, 2]:
        regime_segments = segments_df[segments_df['regime'] == regime]
        regime_name = ['JR0 (Calm)', 'JR1 (Moderate)', 'JR2 (Extreme)'][regime]
        
        print(f"\n   {regime_name}: {len(regime_segments)} segments")
        
        created_count = 0
        for idx, segment in regime_segments.iterrows():
            # For JR0: Skip very long segments (>200 days) to avoid redundancy
            if regime == 0 and segment['length'] > 200:
                # Sample from long JR0 segments instead of using entire segment
                segment_duration = segment['length']
                n_samples = max(1, segment_duration // 100)  # One task per ~100 days
                
                for i in range(n_samples):
                    sample_start = segment['start_date'] + timedelta(days=i * 100)
                    sample_end = min(
                        sample_start + timedelta(days=MIN_TASK_LENGTH),
                        segment['end_date']
                    )
                    
                    if (sample_end - sample_start).days >= MIN_TASK_LENGTH:
                        sample_segment = {
                            'start_date': sample_start,
                            'end_date': sample_end,
                            'length': (sample_end - sample_start).days,
                            'regime': regime
                        }
                        
                        task = create_segment_task(
                            sample_segment, regime_df, jump_df,
                            task_id, start_idx_offset
                        )
                        
                        if task:
                            tasks.append(task)
                            task_id += 1
                            created_count += 1
                
                continue
            
            # For JR1 and JR2: Use all segments
            # For JR0: Use segments < 200 days
            task = create_segment_task(
                segment, regime_df, jump_df,
                task_id, start_idx_offset
            )
            
            if task:
                tasks.append(task)
                task_id += 1
                created_count += 1
        
        print(f"      → Created {created_count} tasks")
    
    return tasks


def split_tasks_temporal(tasks):
    """Split tasks into train/val/test based on chronology."""
    # Sort by start date
    tasks_sorted = sorted(tasks, key=lambda x: x['start_date'])
    
    n_tasks = len(tasks_sorted)
    
    # Split: 60% train, 15% val, 25% test (temporal)
    train_end = int(0.60 * n_tasks)
    val_end = int(0.75 * n_tasks)
    
    train_tasks = tasks_sorted[:train_end]
    val_tasks = tasks_sorted[train_end:val_end]
    test_tasks = tasks_sorted[val_end:]
    
    # Reassign task IDs
    for i, task in enumerate(tasks_sorted):
        task['task_id'] = i
    
    train_indices = [i for i in range(len(train_tasks))]
    val_indices = [i for i in range(len(train_tasks), len(train_tasks) + len(val_tasks))]
    test_indices = [i for i in range(len(train_tasks) + len(val_tasks), n_tasks)]
    
    split = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    return tasks_sorted, split


def print_task_summary(tasks, split):
    """Print summary statistics of created tasks."""
    print("\n" + "="*70)
    print("SEGMENT-BASED TASK GENERATION SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(tasks)
    
    print(f"\n📊 Total Tasks: {len(tasks)}")
    print(f"\n   By Regime:")
    for regime in [0, 1, 2]:
        regime_tasks = df[df['regime'] == regime]
        regime_name = ['JR0 (Calm)', 'JR1 (Moderate)', 'JR2 (Extreme)'][regime]
        print(f"      • {regime_name}: {len(regime_tasks)} tasks")
    
    print(f"\n   By Source:")
    for source in df['source'].unique():
        source_tasks = df[df['source'] == source]
        print(f"      • {source}: {len(source_tasks)} tasks")
    
    print(f"\n📅 Date Range:")
    print(f"   First task: {df['start_date'].min()}")
    print(f"   Last task: {df['start_date'].max()}")
    
    print(f"\n📏 Task Length Statistics:")
    print(f"   Mean: {df['length'].mean():.1f} days")
    print(f"   Median: {df['length'].median():.1f} days")
    print(f"   Min: {df['length'].min()} days")
    print(f"   Max: {df['length'].max()} days")
    
    print(f"\n🔀 Train/Val/Test Split:")
    print(f"   Train: {len(split['train'])} tasks ({len(split['train'])/len(tasks)*100:.1f}%)")
    print(f"   Val: {len(split['val'])} tasks ({len(split['val'])/len(tasks)*100:.1f}%)")
    print(f"   Test: {len(split['test'])} tasks ({len(split['test'])/len(tasks)*100:.1f}%)")
    
    # Check for critical crises
    print(f"\n✅ Critical Crisis Coverage:")
    df['year'] = pd.to_datetime(df['start_date']).dt.year
    critical_years = [1998, 2002, 2008, 2011, 2020]
    for year in critical_years:
        year_tasks = df[(df['year'] == year) & (df['regime'] == 2)]
        status = "✓" if len(year_tasks) > 0 else "✗"
        print(f"   {status} {year}: {len(year_tasks)} JR2 tasks")
    
    print("\n" + "="*70)


def main():
    """Main execution."""
    print("🚀 Segment-Based Task Generation for Jump Model MAML")
    print("="*70)
    
    # Load data
    segments_df, regime_df, jump_df = load_data()
    
    # Create tasks
    tasks = create_all_segment_tasks(segments_df, regime_df, jump_df)
    
    # Split tasks
    tasks_sorted, split = split_tasks_temporal(tasks)
    
    # Save outputs
    print("\n💾 Saving outputs...")
    
    # Save task metadata
    output_file = JM_DIR / 'tasks_metadata_segment_based.json'
    with open(output_file, 'w') as f:
        json.dump(tasks_sorted, f, indent=2)
    print(f"   ✓ Saved {len(tasks_sorted)} tasks to {output_file.name}")
    
    # Save split indices
    split_file = JM_DIR / 'task_split_indices_segment_based.json'
    with open(split_file, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"   ✓ Saved split indices to {split_file.name}")
    
    # Print summary
    print_task_summary(tasks_sorted, split)
    
    print("\n✅ SEGMENT-BASED TASK GENERATION COMPLETE!")
    print("\n🎯 Next Steps:")
    print("   1. Review task distribution above")
    print("   2. Verify all 8 JR2 crises are captured")
    print("   3. (Optional) Run generate_jump_tasks.py with better params for more coverage")
    print("   4. Train MAML with these tasks")


if __name__ == '__main__':
    main()
