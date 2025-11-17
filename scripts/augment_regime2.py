"""
Data Augmentation for Regime 2 (Crisis) Tasks

Addresses data scarcity in VIX>40 crisis periods by creating synthetic tasks
using time-series augmentation techniques.

Usage:
    python scripts/augment_regime2.py --target_count 20 --methods jitter,scale,mixup
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pickle
import json
import argparse
from pathlib import Path
from copy import deepcopy


def jitter_augmentation(features, noise_level=0.01):
    """Add random Gaussian noise to features"""
    noise = np.random.normal(0, noise_level, features.shape)
    return features + noise


def scaling_augmentation(features, scale_range=(0.8, 1.2)):
    """Scale returns by random factor (preserves correlations)"""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    # Only scale return columns (not technical indicators like RSI, MA ratios)
    return_cols = [col for col in range(features.shape[1]) if col < 5]  # First 5 are returns
    
    augmented = features.copy()
    augmented[:, return_cols] *= scale_factor
    return augmented


def time_warp_augmentation(features, warp_range=(0.9, 1.1)):
    """Stretch or compress time dimension"""
    seq_len = features.shape[0]
    warp_factor = np.random.uniform(warp_range[0], warp_range[1])
    new_len = int(seq_len * warp_factor)
    
    # Interpolate to new length
    from scipy.interpolate import interp1d
    old_indices = np.linspace(0, seq_len - 1, seq_len)
    new_indices = np.linspace(0, seq_len - 1, new_len)
    
    augmented = np.zeros((new_len, features.shape[1]))
    for col in range(features.shape[1]):
        f = interp1d(old_indices, features[:, col], kind='linear')
        augmented[:, col] = f(new_indices)
    
    # Truncate or pad to original length
    if new_len > seq_len:
        augmented = augmented[:seq_len]
    else:
        padding = np.tile(augmented[-1:], (seq_len - new_len, 1))
        augmented = np.vstack([augmented, padding])
    
    return augmented


def mixup_augmentation(task1, task2, alpha=0.5):
    """Mix two tasks with random weight"""
    lam = np.random.beta(alpha, alpha)
    
    mixed_task = deepcopy(task1)
    mixed_task['support_x'] = lam * task1['support_x'] + (1 - lam) * task2['support_x']
    mixed_task['support_y'] = lam * task1['support_y'] + (1 - lam) * task2['support_y']
    mixed_task['query_x'] = lam * task1['query_x'] + (1 - lam) * task2['query_x']
    mixed_task['query_y'] = lam * task1['query_y'] + (1 - lam) * task2['query_y']
    
    return mixed_task


def augment_regime2_tasks(dataset, regime2_task_ids, target_count=20, methods=['jitter', 'scale']):
    """
    Augment Regime 2 tasks to reach target count
    
    Args:
        dataset: RegimeTaskDataset instance
        regime2_task_ids: List of existing R2 task IDs
        target_count: Target number of R2 tasks after augmentation
        methods: List of augmentation methods to use
    
    Returns:
        List of augmented tasks (same format as dataset.__getitem__)
    """
    if len(regime2_task_ids) == 0:
        raise ValueError("No Regime 2 tasks found to augment!")
    
    num_existing = len(regime2_task_ids)
    num_to_create = max(0, target_count - num_existing)
    
    print(f"Existing R2 tasks: {num_existing}")
    print(f"Target count: {target_count}")
    print(f"Creating {num_to_create} synthetic tasks...")
    
    augmented_tasks = []
    
    for i in range(num_to_create):
        # Randomly select source task(s)
        if 'mixup' in methods and len(regime2_task_ids) >= 2:
            # Mix two random tasks
            idx1, idx2 = np.random.choice(regime2_task_ids, size=2, replace=False)
            task1 = dataset[idx1]
            task2 = dataset[idx2]
            augmented_task = mixup_augmentation(task1, task2)
        else:
            # Augment single task
            source_idx = np.random.choice(regime2_task_ids)
            task = dataset[source_idx]
            augmented_task = deepcopy(task)
            
            # Apply random augmentation method
            method = np.random.choice([m for m in methods if m != 'mixup'])
            
            if method == 'jitter':
                augmented_task['support_x'] = jitter_augmentation(task['support_x'].numpy())
                augmented_task['query_x'] = jitter_augmentation(task['query_x'].numpy())
            elif method == 'scale':
                augmented_task['support_x'] = scaling_augmentation(task['support_x'].numpy())
                augmented_task['query_x'] = scaling_augmentation(task['query_x'].numpy())
            elif method == 'warp':
                augmented_task['support_x'] = time_warp_augmentation(task['support_x'].numpy())
                augmented_task['query_x'] = time_warp_augmentation(task['query_x'].numpy())
            
            # Convert back to tensors
            import torch
            augmented_task['support_x'] = torch.FloatTensor(augmented_task['support_x'])
            augmented_task['query_x'] = torch.FloatTensor(augmented_task['query_x'])
        
        # Mark as synthetic
        augmented_task['synthetic'] = True
        augmented_task['source_method'] = method if 'mixup' not in methods else 'mixup'
        augmented_task['augmentation_id'] = i
        
        augmented_tasks.append(augmented_task)
    
    return augmented_tasks


def save_augmented_dataset(original_dataset, augmented_tasks, output_path):
    """Save augmented tasks to disk"""
    augmented_data = {
        'original_task_count': len(original_dataset),
        'augmented_task_count': len(augmented_tasks),
        'tasks': augmented_tasks
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(augmented_data, f)
    
    print(f"\n✅ Saved {len(augmented_tasks)} augmented tasks to {output_path}")


def update_split_indices(splits_path, augmented_task_ids, split_type='train'):
    """Update task split indices to include augmented tasks"""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    # Add augmented task IDs to specified split
    splits['task_indices'][f'{split_type}_task_ids'].extend(augmented_task_ids)
    
    # Update regime distribution
    regime_key = f'regime_2'
    splits['regime_distribution'][split_type][regime_key] += len(augmented_task_ids)
    
    # Update task counts
    splits['task_counts'][split_type] += len(augmented_task_ids)
    splits['task_counts']['total'] += len(augmented_task_ids)
    
    # Add augmentation metadata
    splits['augmentation_info'] = {
        'augmented_count': len(augmented_task_ids),
        'target_regime': 'regime_2',
        'methods_used': 'jitter,scale,mixup',
        'augmented_at': pd.Timestamp.now().isoformat()
    }
    
    # Save updated splits
    output_path = splits_path.replace('.json', '_augmented.json')
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✅ Updated task splits saved to {output_path}")
    return output_path


def main(args):
    print("=" * 60)
    print("Regime 2 Data Augmentation")
    print("=" * 60)
    
    # Load dataset
    from src.data.task_dataset import RegimeTaskDataset
    
    print("\nLoading dataset...")
    with open(f'{args.data_dir}/regime_segments_ma_vix_3regime.pkl', 'rb') as f:
        segments = pickle.load(f)
    
    features_df = pd.read_csv(
        f'{args.data_dir}/features_imputed_with_targets.csv',
        index_col=0,
        parse_dates=True
    )
    
    dataset = RegimeTaskDataset(
        segments=segments,
        features_df=features_df,
        support_size=30,
        query_size=10,
        stride=40,
        regime2_support_size=10,
        regime2_stride=20
    )
    
    # Load task splits
    with open(f'{args.data_dir}/task_split_indices_v2_no_leakage.json', 'r') as f:
        splits = json.load(f)
    
    # Find Regime 2 tasks in training set
    train_ids = splits['task_indices']['train_task_ids']
    regime2_train_ids = []
    
    for task_id in train_ids:
        task = dataset[task_id]
        if task['regime_label'] == 2:
            regime2_train_ids.append(task_id)
    
    print(f"\nFound {len(regime2_train_ids)} Regime 2 tasks in training set")
    print(f"Task IDs: {regime2_train_ids}")
    
    # Augment
    methods = args.methods.split(',')
    print(f"\nUsing augmentation methods: {methods}")
    
    augmented_tasks = augment_regime2_tasks(
        dataset=dataset,
        regime2_task_ids=regime2_train_ids,
        target_count=args.target_count,
        methods=methods
    )
    
    # Save augmented tasks
    output_path = f'{args.output_dir}/augmented_regime2_tasks.pkl'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_augmented_dataset(dataset, augmented_tasks, output_path)
    
    # Update splits
    augmented_task_ids = list(range(
        len(dataset),
        len(dataset) + len(augmented_tasks)
    ))
    
    new_splits_path = update_split_indices(
        f'{args.data_dir}/task_split_indices_v2_no_leakage.json',
        augmented_task_ids,
        split_type='train'
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original R2 training tasks: {len(regime2_train_ids)}")
    print(f"Augmented tasks created: {len(augmented_tasks)}")
    print(f"New R2 training tasks: {len(regime2_train_ids) + len(augmented_tasks)}")
    print(f"\nFiles created:")
    print(f"  - {output_path}")
    print(f"  - {new_splits_path}")
    print("\n⚠️  Update your training script to load augmented tasks!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment Regime 2 tasks')
    
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='data/augmented')
    parser.add_argument('--target_count', type=int, default=20,
                        help='Target number of R2 tasks after augmentation')
    parser.add_argument('--methods', type=str, default='jitter,scale,mixup',
                        help='Comma-separated augmentation methods: jitter,scale,warp,mixup')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    main(args)
