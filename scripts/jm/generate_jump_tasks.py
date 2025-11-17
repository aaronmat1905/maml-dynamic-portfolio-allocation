"""
Generate MAML Tasks from Jump Regimes

Creates meta-learning task splits from jump-based regime classification.
Each task is a time window with jump characteristics (intensity, size, clustering).

Output:
- task_split_indices_jm.json: Train/val/test task splits
- Jump-specific features added to task dataset

Usage:
    python scripts/jm/generate_jump_tasks.py --segments data/jm/jump_regime_segments.csv
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import timedelta


def load_data():
    """Load jump regime labels and feature data."""
    # Load jump regime labels
    regime_path = Path('data/jm/jump_regime_labels.csv')
    print(f"Loading regime labels from {regime_path}...")
    regime_df = pd.read_csv(regime_path, parse_dates=['Date'], index_col='Date')
    regime_df.columns = regime_df.columns.str.lower().str.replace(' ', '_')
    
    # Load features (reuse from HMM pipeline)
    features_path = Path('data/processed/features_imputed_with_targets.csv')
    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date')
    features_df.columns = features_df.columns.str.lower().str.replace(' ', '_')
    
    return regime_df, features_df


def create_jump_tasks(regime_df, features_df, min_task_length=40, stride=20):
    """
    Create MAML tasks from jump regime segments.
    
    Args:
        regime_df: DataFrame with regime labels and jump metrics
        features_df: DataFrame with engineered features
        min_task_length: Minimum days for a task (default 40)
        stride: Days between task start dates (default 20)
        
    Returns:
        List of task dicts with metadata
    """
    tasks = []
    task_id = 0
    
    # Standardize column names
    regime_df.columns = regime_df.columns.str.lower().str.replace(' ', '_')
    
    # Merge regime labels with features
    # Find common columns to avoid duplicates
    common_cols = list(set(regime_df.columns) & set(features_df.columns))
    regime_cols_to_join = ['regime_label', 'jump_intensity', 'avg_jump_size', 
                           'jump_clustering', 'negative_jump_ratio']
    regime_cols_to_join = [col for col in regime_cols_to_join if col in regime_df.columns]
    
    print(f"Joining regime columns: {regime_cols_to_join}")
    print(f"Features shape: {features_df.shape}, Regime shape: {regime_df.shape}")
    
    merged_df = features_df.join(regime_df[regime_cols_to_join], how='inner')
    
    print(f"Merged shape: {merged_df.shape}")
    print(f"Merged date range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    if len(merged_df) == 0:
        print("ERROR: No data after merge! Check date alignment.")
        return tasks
    
    # Create rolling windows as tasks
    start_date = merged_df.index[0]  # Use index position, not min()
    end_date = merged_df.index[-1]
    max_start_date = end_date - pd.Timedelta(days=min_task_length)
    
    print(f"Creating tasks from {start_date} to {max_start_date}")
    print(f"Expected ~{int((end_date - start_date).days / stride)} tasks")
    
    current_date = start_date
    iteration = 0
    while current_date <= max_start_date:
        task_end = current_date + pd.Timedelta(days=min_task_length)
        
        # Get data for this window
        task_df = merged_df[(merged_df.index >= current_date) & (merged_df.index < task_end)]
        
        iteration += 1
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}: current_date={current_date.date()}, task_df length={len(task_df)}")
        
        # Relaxed condition: accept tasks with at least 75% of target length (to account for weekends/holidays)
        min_required = int(min_task_length * 0.75)
        if len(task_df) >= min_required:
            # Determine dominant regime in this window
            regime_counts = task_df['regime_label'].value_counts()
            dominant_regime = regime_counts.idxmax()
            regime_purity = regime_counts.max() / len(task_df)
            
            # Calculate jump characteristics for this task
            jump_freq = task_df['jump_intensity'].mean()
            avg_jump_sz = task_df['avg_jump_size'].mean()
            jump_cluster = task_df['jump_clustering'].mean()
            neg_jump_ratio = task_df['negative_jump_ratio'].mean()
            
            # Get VIX info
            vix_col = None
            for col in ['vix_close', 'vix_adj_close']:
                if col in task_df.columns:
                    vix_col = col
                    break
            
            # Task metadata
            task = {
                'task_id': task_id,
                'regime': int(dominant_regime),
                'regime_purity': float(regime_purity),
                'start_date': current_date.strftime('%Y-%m-%d'),
                'end_date': task_end.strftime('%Y-%m-%d'),
                'length': len(task_df),
                'start_idx': int(merged_df.index.get_loc(current_date)),
                'end_idx': int(merged_df.index.get_loc(task_df.index[-1])) + 1,
                # Jump characteristics
                'jump_frequency': float(jump_freq),
                'avg_jump_size': float(avg_jump_sz),
                'jump_clustering': float(jump_cluster),
                'negative_jump_ratio': float(neg_jump_ratio),
                # VIX for comparison with HMM
                'max_vix': float(task_df[vix_col].max()) if vix_col else None,
                'avg_vix': float(task_df[vix_col].mean()) if vix_col else None,
            }
            
            tasks.append(task)
            task_id += 1
        
        # Move to next window
        current_date += pd.Timedelta(days=stride)
    
    print(f"\nTotal iterations: {iteration}, Tasks created: {len(tasks)}")
    return tasks


def split_tasks(tasks, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, 
                holdout_crisis='2022_ukraine'):
    """
    Split tasks into train/val/test with stratification by regime.
    
    Args:
        tasks: List of task dicts
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        holdout_crisis: Optional specific crisis to hold out for testing
        
    Returns:
        Dict with train/val/test task IDs
    """
    # Separate by regime
    tasks_by_regime = {0: [], 1: [], 2: []}
    for task in tasks:
        tasks_by_regime[task['regime']].append(task)
    
    # Hold out Ukraine 2022 if specified
    holdout_tasks = []
    if holdout_crisis == '2022_ukraine':
        ukraine_start = pd.Timestamp('2022-02-15')
        ukraine_end = pd.Timestamp('2022-04-30')
        
        for regime in tasks_by_regime:
            regime_tasks = tasks_by_regime[regime]
            # Separate Ukraine tasks
            ukraine = [t for t in regime_tasks if 
                      pd.Timestamp(t['start_date']) >= ukraine_start and 
                      pd.Timestamp(t['start_date']) <= ukraine_end]
            non_ukraine = [t for t in regime_tasks if 
                          not (pd.Timestamp(t['start_date']) >= ukraine_start and 
                               pd.Timestamp(t['start_date']) <= ukraine_end)]
            
            tasks_by_regime[regime] = non_ukraine
            holdout_tasks.extend(ukraine)
    
    # Split each regime proportionally
    train_ids, val_ids, test_ids = [], [], []
    
    for regime, regime_tasks in tasks_by_regime.items():
        n = len(regime_tasks)
        
        # Sort by date for chronological splitting
        regime_tasks_sorted = sorted(regime_tasks, key=lambda x: x['start_date'])
        
        # Calculate split points
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Split
        train_tasks = regime_tasks_sorted[:n_train]
        val_tasks = regime_tasks_sorted[n_train:n_train + n_val]
        test_tasks = regime_tasks_sorted[n_train + n_val:]
        
        train_ids.extend([t['task_id'] for t in train_tasks])
        val_ids.extend([t['task_id'] for t in val_tasks])
        test_ids.extend([t['task_id'] for t in test_tasks])
        
        print(f"\nRegime {regime} split:")
        print(f"  Total: {n} tasks")
        print(f"  Train: {len(train_tasks)} tasks")
        print(f"  Val: {len(val_tasks)} tasks")
        print(f"  Test: {len(test_tasks)} tasks")
    
    # Add holdout tasks to test set
    if holdout_tasks:
        holdout_ids = [t['task_id'] for t in holdout_tasks]
        test_ids.extend(holdout_ids)
        print(f"\nHoldout (Ukraine 2022):")
        print(f"  Test: {len(holdout_tasks)} tasks (added to test set)")
    
    return {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
        'test': sorted(test_ids),
        'holdout_ukraine': sorted([t['task_id'] for t in holdout_tasks]) if holdout_tasks else []
    }


def compare_with_hmm_tasks(jm_tasks, hmm_path='data/processed/task_split_indices_v2_no_leakage.json'):
    """Compare JM tasks with HMM tasks."""
    print("\n" + "="*70)
    print("COMPARISON: JUMP MODEL vs HMM TASKS")
    print("="*70)
    
    # Load HMM tasks if available
    if Path(hmm_path).exists():
        with open(hmm_path, 'r') as f:
            hmm_splits = json.load(f)
        
        print("\nHMM (VIX threshold) tasks:")
        print(f"  Train: {len(hmm_splits.get('train', []))} tasks")
        print(f"  Val: {len(hmm_splits.get('val', []))} tasks")
        print(f"  Test: {len(hmm_splits.get('test', []))} tasks")
    else:
        print("\nHMM tasks file not found")
    
    # JM tasks by regime
    print("\nJump Model tasks:")
    print(f"  Total: {len(jm_tasks)} tasks")
    
    regime_counts = {0: 0, 1: 0, 2: 0}
    for task in jm_tasks:
        regime_counts[task['regime']] += 1
    
    print(f"  JR0 (Calm): {regime_counts[0]} tasks")
    print(f"  JR1 (Moderate): {regime_counts[1]} tasks")
    print(f"  JR2 (Extreme): {regime_counts[2]} tasks")
    
    # JR2 crisis tasks
    jr2_tasks = [t for t in jm_tasks if t['regime'] == 2]
    print(f"\nJR2 (Crisis) task breakdown:")
    print(f"  Total JR2 tasks: {len(jr2_tasks)}")
    print(f"  Avg jump frequency: {np.mean([t['jump_frequency'] for t in jr2_tasks]):.2%}")
    print(f"  Avg jump size: {np.mean([t['avg_jump_size'] for t in jr2_tasks]):.2%}")
    
    # Show some example JR2 tasks
    print(f"\nExample JR2 tasks (first 10):")
    for i, task in enumerate(jr2_tasks[:10], 1):
        print(f"  {i}. {task['start_date']} to {task['end_date']} "
              f"(jump_freq={task['jump_frequency']:.1%}, jump_size={task['avg_jump_size']:.2%})")
    
    # Check Ukraine 2022
    ukraine_tasks = [t for t in jm_tasks if '2022-02' <= t['start_date'] <= '2022-04']
    if ukraine_tasks:
        print(f"\n✅ Ukraine 2022 tasks found: {len(ukraine_tasks)}")
        for task in ukraine_tasks:
            regime_name = {0: 'JR0 (Calm)', 1: 'JR1 (Moderate)', 2: 'JR2 (Extreme)'}[task['regime']]
            print(f"   {task['start_date']} to {task['end_date']}: {regime_name} "
                  f"(jump_freq={task['jump_frequency']:.1%})")
    else:
        print("\n❌ No Ukraine 2022 tasks found")


def main():
    parser = argparse.ArgumentParser(description='Generate MAML tasks from jump regimes')
    parser.add_argument('--segments', type=str, default='data/jm/jump_regime_segments.csv',
                       help='Path to regime segments CSV')
    parser.add_argument('--min_task_length', type=int, default=40,
                       help='Minimum task length in days (default 40)')
    parser.add_argument('--stride', type=int, default=20,
                       help='Days between task starts (default 20)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Training set proportion (default 0.6)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set proportion (default 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set proportion (default 0.2)')
    parser.add_argument('--holdout_ukraine', action='store_true',
                       help='Hold out Ukraine 2022 for testing (recommended)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("JUMP MODEL TASK GENERATION FOR MAML")
    print("="*70)
    
    # Load data
    regime_df, features_df = load_data()
    
    print(f"\nData loaded:")
    print(f"  Regime labels: {len(regime_df)} days")
    print(f"  Features: {len(features_df)} days")
    print(f"  Date range: {regime_df.index.min()} to {regime_df.index.max()}")
    
    # Create tasks
    print(f"\nCreating tasks (length={args.min_task_length}, stride={args.stride})...")
    tasks = create_jump_tasks(regime_df, features_df, 
                             min_task_length=args.min_task_length,
                             stride=args.stride)
    
    print(f"\nTotal tasks created: {len(tasks)}")
    
    # Split tasks
    print(f"\nSplitting tasks (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})...")
    holdout = '2022_ukraine' if args.holdout_ukraine else None
    task_splits = split_tasks(tasks, 
                             train_ratio=args.train_ratio,
                             val_ratio=args.val_ratio,
                             test_ratio=args.test_ratio,
                             holdout_crisis=holdout)
    
    print(f"\nFinal splits:")
    print(f"  Train: {len(task_splits['train'])} tasks")
    print(f"  Val: {len(task_splits['val'])} tasks")
    print(f"  Test: {len(task_splits['test'])} tasks")
    if task_splits.get('holdout_ukraine'):
        print(f"  Holdout (Ukraine): {len(task_splits['holdout_ukraine'])} tasks")
    
    # Save task metadata
    tasks_metadata_path = Path('data/jm/tasks_metadata.json')
    with open(tasks_metadata_path, 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"\n✅ Saved task metadata to {tasks_metadata_path}")
    
    # Save splits
    splits_path = Path('data/jm/task_split_indices_jm.json')
    with open(splits_path, 'w') as f:
        json.dump(task_splits, f, indent=2)
    print(f"✅ Saved task splits to {splits_path}")
    
    # Comparison with HMM
    compare_with_hmm_tasks(tasks)
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    jr2_count = len([t for t in tasks if t['regime'] == 2])
    jr2_train = len([tid for tid in task_splits['train'] if tasks[tid]['regime'] == 2])
    
    print(f"JM JR2 (crisis) tasks: {jr2_count} total, {jr2_train} for training")
    print(f"HMM R2 tasks: ~5 original (from 3 crisis periods)")
    print(f"JM provides {jr2_count/5:.1f}x more crisis task diversity!")
    
    if task_splits.get('holdout_ukraine'):
        print(f"\n✅ Ukraine 2022 held out as pure test crisis (unseen during training)")
        print(f"   This enables testing generalization to novel geopolitical crises")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review tasks_metadata.json - check jump characteristics")
    print("2. Verify Ukraine 2022 tasks exist and are marked as holdout")
    print("3. Run: python scripts/jm/train_maml_jm.py --use_jump_tasks")
    print("4. Compare JM-MAML vs HMM-MAML on Ukraine 2022 test set")


if __name__ == '__main__':
    main()
