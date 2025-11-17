"""
Supervised Learning Baseline for Portfolio Allocation

Trains a standard model on pooled data (no meta-learning)
Used to compare against MAML performance

Usage:
    python scripts/train_baseline.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.data.task_dataset import RegimeTaskDataset
from src.models.portfolio_net import PortfolioNet


def load_dataset(data_dir='data/processed'):
    """Load pre-processed dataset"""
    with open(f'{data_dir}/regime_segments_ma_vix_3regime.pkl', 'rb') as f:
        segments = pickle.load(f)
    
    import pandas as pd
    features_df = pd.read_csv(
        f'{data_dir}/features_imputed_with_targets.csv',
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
    
    with open(f'{data_dir}/task_split_indices_v2_no_leakage.json', 'r') as f:
        splits = json.load(f)
    
    train_ids = splits['task_indices']['train_task_ids']
    val_ids = splits['task_indices']['val_task_ids']
    test_ids = splits['task_indices']['test_task_ids']
    
    return dataset, train_ids, val_ids, test_ids


def pool_tasks(dataset, task_ids):
    """Pool all tasks into single dataset (support + query combined)"""
    all_x = []
    all_y = []
    
    for task_id in task_ids:
        task = dataset[task_id]
        
        # Combine support and query
        support_x = task['support_x']
        support_y = task['support_y']
        query_x = task['query_x']
        query_y = task['query_y']
        
        all_x.append(support_x)
        all_x.append(query_x)
        all_y.append(support_y)
        all_y.append(query_y)
    
    # Concatenate all
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    return all_x, all_y


def train_supervised(model, train_x, train_y, val_x, val_y, args):
    """Standard supervised training"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Move data to device
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting supervised training...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Shuffle data
        indices = torch.randperm(len(train_x))
        
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i+args.batch_size]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        train_loss = np.mean(epoch_losses)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = criterion(val_preds, val_y).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
    
    return train_losses, val_losses, best_val_loss


def evaluate_on_tasks(model, dataset, test_ids, device):
    """Evaluate supervised model on test tasks (no adaptation)"""
    model.eval()
    criterion = nn.MSELoss()
    
    all_losses = []
    regime_losses = {0: [], 1: [], 2: []}
    
    with torch.no_grad():
        for task_id in test_ids:
            task = dataset[task_id]
            query_x = task['query_x'].to(device)
            query_y = task['query_y'].to(device)
            regime = task['regime_label']
            
            # Zero-shot evaluation (no adaptation)
            preds = model(query_x)
            loss = criterion(preds, query_y).item()
            
            all_losses.append(loss)
            regime_losses[regime].append(loss)
    
    avg_loss = np.mean(all_losses)
    regime_avg = {
        r: np.mean(losses) if losses else float('nan')
        for r, losses in regime_losses.items()
    }
    
    return avg_loss, regime_avg


def main(args):
    print("=" * 60)
    print("Supervised Baseline Training")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset, train_ids, val_ids, test_ids = load_dataset(args.data_dir)
    
    print(f"Train tasks: {len(train_ids)}")
    print(f"Val tasks: {len(val_ids)}")
    print(f"Test tasks: {len(test_ids)}")
    
    # Pool tasks into single dataset
    print("\nPooling tasks into single dataset...")
    train_x, train_y = pool_tasks(dataset, train_ids)
    val_x, val_y = pool_tasks(dataset, val_ids)
    
    print(f"Train samples: {len(train_x)}")
    print(f"Val samples: {len(val_x)}")
    
    # Create model
    model = PortfolioNet(
        input_size=len(dataset.feature_cols),
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_losses, val_losses, best_val_loss = train_supervised(
        model, train_x, train_y, val_x, val_y, args
    )
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth"))
    model = model.to(device)
    
    test_loss, regime_losses = evaluate_on_tasks(model, dataset, test_ids, device)
    
    print(f"\nOverall Test MSE: {test_loss:.6f}")
    print("\nPer-Regime Test MSE:")
    print(f"  Regime 0 (Bearish): {regime_losses[0]:.6f}")
    print(f"  Regime 1 (Bullish): {regime_losses[1]:.6f}")
    print(f"  Regime 2 (Crisis): {regime_losses[2]:.6f}")
    
    # Save results
    results = {
        'config': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'regime_losses': regime_losses,
        'best_val_loss': best_val_loss
    }
    
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supervised Baseline')
    
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='experiments/supervised_baseline')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
