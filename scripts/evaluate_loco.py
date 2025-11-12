"""
Leave-One-Crisis-Out (LOCO) Evaluation for MAML

Tests whether the model can generalize to UNSEEN crisis types by:
1. Holding out one crisis period
2. Training on remaining crises (+ augmented data)
3. Testing on the held-out crisis

This proves the model learns general crisis adaptation, not just memorization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
import json
import argparse
from pathlib import Path

from scripts.train_maml import MAMLTrainer, load_dataset
from src.models.portfolio_net import PortfolioNet
from src.data.task_dataset import RegimeTaskDataset


def get_crisis_segments():
    """Return the 3 crisis periods in the dataset"""
    return {
        'crisis_2008': {
            'name': '2008 Financial Crisis',
            'start_date': '2008-10-02',
            'end_date': '2008-12-30'
        },
        'crisis_2009': {
            'name': '2009 Continued Crisis',  
            'start_date': '2009-01-29',
            'end_date': '2009-04-02'
        },
        'crisis_2020': {
            'name': '2020 COVID Crash',
            'start_date': '2020-03-06',
            'end_date': '2020-04-13'
        }
    }


def filter_tasks_by_crisis(dataset, task_ids, holdout_crisis):
    """
    Split tasks into train/test based on which crisis to hold out
    
    Args:
        dataset: RegimeTaskDataset instance
        task_ids: List of task IDs to filter
        holdout_crisis: Dict with 'start_date' and 'end_date'
        
    Returns:
        train_tasks, test_tasks
    """
    import pandas as pd
    
    holdout_start = pd.Timestamp(holdout_crisis['start_date'])
    holdout_end = pd.Timestamp(holdout_crisis['end_date'])
    
    train_tasks = []
    test_tasks = []
    
    for task_id in task_ids:
        task = dataset[task_id]
        
        # Check if task overlaps with holdout crisis
        task_start_idx = dataset.tasks[task_id]['start_idx']
        task_dates = dataset.features_df.index[task_start_idx:task_start_idx + dataset.window_size]
        
        if len(task_dates) == 0:
            continue
            
        task_start = task_dates[0]
        task_end = task_dates[-1]
        
        # Check overlap with holdout crisis
        overlaps = (task_start <= holdout_end) and (task_end >= holdout_start)
        
        if overlaps:
            test_tasks.append(task)
        else:
            train_tasks.append(task)
    
    return train_tasks, test_tasks


def run_loco_evaluation(args):
    """Run Leave-One-Crisis-Out evaluation"""
    print("=" * 60)
    print("Leave-One-Crisis-Out (LOCO) Evaluation")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load full dataset
    train_tasks, val_tasks, test_tasks, dataset = load_dataset(
        args.data_dir,
        use_augmented=True,
        augmented_file='data/augmented/augmented_regime2_tasks.pkl'
    )
    
    crises = get_crisis_segments()
    results = {}
    
    # Run LOCO for each crisis
    for crisis_key, crisis_info in crises.items():
        print(f"\n{'='*60}")
        print(f"Holdout Crisis: {crisis_info['name']}")
        print(f"Period: {crisis_info['start_date']} to {crisis_info['end_date']}")
        print(f"{'='*60}")
        
        # Get R2 tasks only
        all_r2_tasks = [t for t in train_tasks + test_tasks if t['regime_label'] == 2]
        
        # Split into train/test based on holdout
        loco_train, loco_test = filter_tasks_by_crisis(dataset, 
                                                         range(len(dataset)),
                                                         crisis_info)
        
        # Filter to R2 only
        loco_train = [t for t in loco_train if t['regime_label'] == 2]
        loco_test = [t for t in loco_test if t['regime_label'] == 2]
        
        print(f"\nLOCO Split:")
        print(f"  Train R2 tasks: {len(loco_train)}")
        print(f"  Test R2 tasks: {len(loco_test)}")
        
        if len(loco_test) == 0:
            print(f"  ⚠️  No test tasks for this crisis, skipping...")
            continue
        
        # Create model
        model = PortfolioNet(
            input_size=len(dataset.feature_cols),
            hidden_size=args.hidden_size
        )
        
        # Create trainer
        trainer = MAMLTrainer(
            model=model,
            inner_lr=args.inner_lr,
            outer_lr=args.outer_lr,
            inner_steps=args.inner_steps,
            device=device
        )
        
        # Quick training (10 epochs for LOCO)
        print(f"\nTraining (10 epochs)...")
        for epoch in range(1, 11):
            train_loss = trainer.train_epoch(loco_train, meta_batch_size=4)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}/10 | Loss: {train_loss:.6f}")
        
        # Evaluate on held-out crisis
        test_loss = trainer.evaluate(loco_test)
        
        print(f"\n✓ Held-out {crisis_info['name']} MSE: {test_loss:.6f}")
        
        results[crisis_key] = {
            'name': crisis_info['name'],
            'test_mse': test_loss,
            'train_tasks': len(loco_train),
            'test_tasks': len(loco_test)
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("LOCO EVALUATION SUMMARY")
    print("=" * 60)
    
    for crisis_key, res in results.items():
        print(f"\n{res['name']}:")
        print(f"  Test MSE: {res['test_mse']:.6f}")
        print(f"  Train tasks: {res['train_tasks']}, Test tasks: {res['test_tasks']}")
    
    avg_mse = np.mean([r['test_mse'] for r in results.values()])
    print(f"\n{'='*60}")
    print(f"Average Held-Out Crisis MSE: {avg_mse:.6f}")
    print(f"{'='*60}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(f"{output_dir}/loco_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}/loco_results.json")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Leave-One-Crisis-Out Evaluation')
    
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='experiments/loco_evaluation')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--outer_lr', type=float, default=0.0001)
    parser.add_argument('--inner_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    run_loco_evaluation(args)
