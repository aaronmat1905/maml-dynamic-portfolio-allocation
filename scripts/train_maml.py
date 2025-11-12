"""
MAML Training Script for Portfolio Allocation

Optimized for Google Colab (CPU-friendly, small dataset)
Implements Model-Agnostic Meta-Learning (MAML) for regime-aware portfolio prediction

Usage:
    python scripts/train_maml.py --config configs/experiment.yaml
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
import matplotlib.pyplot as plt

# Import your dataset and model
from src.data.task_dataset import RegimeTaskDataset
from src.models.portfolio_net import PortfolioNet


class MAMLTrainer:
    """
    MAML Trainer for Portfolio Prediction
    
    Simple implementation without external dependencies (learn2learn)
    Perfect for understanding the algorithm and CPU training
    """
    
    def __init__(
        self,
        model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        device='cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        # MAML hyperparameters
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        # Meta-optimizer (outer loop)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def inner_loop(self, support_x, support_y):
        """
        Inner loop: Adapt model to support set
        
        Args:
            support_x: (support_size, seq_len, features)
            support_y: (support_size, 1)
            
        Returns:
            adapted_params: Adapted model parameters after inner steps
        """
        # Clone model parameters for task-specific adaptation
        temp_model = PortfolioNet(
            input_size=self.model.input_size,
            hidden_size=self.model.hidden_size
        ).to(self.device)
        temp_model.load_state_dict(self.model.state_dict())
        
        # Inner optimizer (SGD for fast adaptation)
        inner_optimizer = optim.SGD(temp_model.parameters(), lr=self.inner_lr)
        
        # Adapt on support set
        temp_model.train()
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass
            preds = temp_model(support_x)
            loss = self.criterion(preds, support_y)
            
            # Backward pass
            loss.backward()
            inner_optimizer.step()
        
        return temp_model
    
    def outer_loop(self, task_batch):
        """
        Outer loop: Meta-update based on query set performance
        
        Args:
            task_batch: List of tasks, each with support/query sets
            
        Returns:
            meta_loss: Average loss across query sets
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        valid_tasks = 0
        
        for task in task_batch:
            # Extract task data
            support_x = task['support_x'].to(self.device)
            support_y = task['support_y'].to(self.device)
            query_x = task['query_x'].to(self.device)
            query_y = task['query_y'].to(self.device)
            
            # Inner loop: adapt to support set
            adapted_model = self.inner_loop(support_x, support_y)
            
            # Evaluate adapted model on query set
            adapted_model.eval()
            query_preds = adapted_model(query_x)
            task_loss = self.criterion(query_preds, query_y)
            
            meta_loss += task_loss
            valid_tasks += 1
        
        if valid_tasks == 0:
            return 0.0
        
        # Average meta-loss
        meta_loss = meta_loss / valid_tasks
        
        # Meta-optimization step
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def train_epoch(self, train_tasks, meta_batch_size=8):
        """Train one epoch"""
        self.model.train()
        
        # Shuffle tasks
        indices = np.random.permutation(len(train_tasks))
        
        epoch_losses = []
        
        # Process in meta-batches
        for i in range(0, len(indices), meta_batch_size):
            batch_indices = indices[i:i+meta_batch_size]
            task_batch = [train_tasks[idx] for idx in batch_indices]
            
            # MAML update
            meta_loss = self.outer_loop(task_batch)
            epoch_losses.append(meta_loss)
        
        return np.mean(epoch_losses)
    
    def evaluate(self, test_tasks, return_per_regime=False):
        """
        Evaluate on test tasks
        
        Args:
            test_tasks: List of test tasks
            return_per_regime: If True, return losses per regime
            
        Returns:
            avg_loss: Average MSE across all test tasks
            regime_losses: (optional) Dict of per-regime losses
        """
        self.model.eval()
        
        all_losses = []
        regime_losses = {0: [], 1: [], 2: []}
        
        for task in test_tasks:
            support_x = task['support_x'].to(self.device)
            support_y = task['support_y'].to(self.device)
            query_x = task['query_x'].to(self.device)
            query_y = task['query_y'].to(self.device)
            regime = task['regime_label']
            
            # Adapt on support set (inner loop) - needs gradients
            adapted_model = self.inner_loop(support_x, support_y)
            
            # Evaluate on query set (no gradients needed)
            adapted_model.eval()
            with torch.no_grad():
                query_preds = adapted_model(query_x)
                task_loss = self.criterion(query_preds, query_y).item()
            
            all_losses.append(task_loss)
            regime_losses[regime].append(task_loss)
        
        avg_loss = np.mean(all_losses)
        
        if return_per_regime:
            regime_avg = {
                r: np.mean(losses) if losses else float('nan')
                for r, losses in regime_losses.items()
            }
            return avg_loss, regime_avg
        
        return avg_loss


def load_dataset(data_dir='data/processed', use_augmented=False, augmented_file=None):
    """Load pre-processed dataset and splits"""
    # Load segments
    with open(f'{data_dir}/regime_segments_ma_vix_3regime.pkl', 'rb') as f:
        segments = pickle.load(f)
    
    # Load features
    import pandas as pd
    features_df = pd.read_csv(
        f'{data_dir}/features_imputed_with_targets.csv',
        index_col=0,
        parse_dates=True
    )
    
    # Create dataset
    dataset = RegimeTaskDataset(
        segments=segments,
        features_df=features_df,
        support_size=30,
        query_size=10,
        stride=40,
        regime2_support_size=10,
        regime2_stride=20
    )
    
    # Load splits
    split_file = 'task_split_indices_v2_no_leakage_augmented.json' if use_augmented else 'task_split_indices_v2_no_leakage.json'
    with open(f'{data_dir}/{split_file}', 'r') as f:
        splits = json.load(f)
    
    train_ids = splits['task_indices']['train_task_ids']
    val_ids = splits['task_indices']['val_task_ids']
    test_ids = splits['task_indices']['test_task_ids']
    
    # Create task lists
    train_tasks = [dataset[i] for i in train_ids if i < len(dataset)]
    val_tasks = [dataset[i] for i in val_ids]
    test_tasks = [dataset[i] for i in test_ids]
    
    # Add augmented tasks if specified
    if use_augmented and augmented_file:
        print(f"\nLoading augmented tasks from {augmented_file}...")
        with open(augmented_file, 'rb') as f:
            augmented_data = pickle.load(f)
        
        augmented_tasks = augmented_data['tasks']
        print(f"Loaded {len(augmented_tasks)} augmented tasks")
        
        # Add to training set
        train_tasks.extend(augmented_tasks)
        print(f"Total training tasks after augmentation: {len(train_tasks)}")
    
    return train_tasks, val_tasks, test_tasks, dataset


def main(args):
    """Main training function"""
    print("=" * 60)
    print("MAML Training for Portfolio Allocation")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_tasks, val_tasks, test_tasks, dataset = load_dataset(
        args.data_dir, 
        use_augmented=args.use_augmented,
        augmented_file=args.augmented_file
    )
    
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks: {len(val_tasks)}")
    print(f"Test tasks: {len(test_tasks)}")
    print(f"Features: {len(dataset.feature_cols)}")
    
    # Create model
    print("\nCreating model...")
    model = PortfolioNet(
        input_size=len(dataset.feature_cols),
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = MAMLTrainer(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        device=device
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = trainer.train_epoch(train_tasks, args.meta_batch_size)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = trainer.evaluate(val_tasks)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
            print(f"  ✓ New best model saved!")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth"))
    trainer.model = model
    
    # Evaluate with per-regime breakdown
    test_loss, regime_losses = trainer.evaluate(test_tasks, return_per_regime=True)
    
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
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MAML Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.output_dir}/training_curves.png", dpi=150)
    print(f"\n✅ Results saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MAML for Portfolio Allocation')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/maml_baseline',
                        help='Output directory for results')
    parser.add_argument('--use_augmented', action='store_true',
                        help='Use augmented Regime 2 tasks')
    parser.add_argument('--augmented_file', type=str, default='data/augmented/augmented_regime2_tasks.pkl',
                        help='Path to augmented tasks file')
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # MAML hyperparameters
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='Outer loop (meta) learning rate')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='Number of inner loop gradient steps')
    parser.add_argument('--meta_batch_size', type=int, default=8,
                        help='Number of tasks per meta-batch')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
