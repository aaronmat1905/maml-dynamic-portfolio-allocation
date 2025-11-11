"""
Regime Task Dataset for MAML Meta-Learning

PyTorch Dataset for creating meta-learning tasks from regime-segmented market data.
Uses adaptive window sizes based on regime characteristics.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class RegimeTaskDataset(Dataset):
    """
    PyTorch Dataset for MAML meta-learning tasks from MA + VIX 3-regime detection.
    
    Each task contains:
    - Support set: 10-30 consecutive days (for adaptation)
    - Query set: 10 consecutive days (for evaluation)
    - All within a single regime segment
    
    **3 Regimes with Adaptive Windows:**
    - Regime 0/1: 30 support + 10 query = 40 days
    - Regime 2: 10 support + 10 query = 20 days (VIX spikes are short-lived)
    """
    
    def __init__(self, segments, features_df, support_size=30, query_size=10, stride=40,
                 regime2_support_size=10, regime2_stride=20):
        """
        Args:
            segments: List of segment dicts from MA + VIX 3-regime detection
            features_df: DataFrame with features and targets (with _target suffix)
            support_size: Number of days in support set for Regime 0/1 (default: 30)
            query_size: Number of days in query set (default: 10)
            stride: Sliding window stride for Regime 0/1 (default: 40)
            regime2_support_size: Smaller support for Regime 2 (default: 10)
            regime2_stride: Smaller stride for Regime 2 (default: 20)
        """
        self.segments = segments
        self.features_df = features_df
        self.support_size = support_size
        self.query_size = query_size
        self.stride = stride
        self.regime2_support_size = regime2_support_size
        self.regime2_stride = regime2_stride
        self.window_size = support_size + query_size
        self.regime2_window_size = regime2_support_size + query_size
        
        # Identify feature and target columns (exclude imputation flags)
        self.feature_cols = [
            col for col in features_df.columns 
            if not col.endswith('_target') and not col.startswith('imputed_')
        ]
        self.target_cols = [col for col in features_df.columns if col.endswith('_target')]
        
        # Generate all tasks
        self.tasks = self._generate_tasks()
        
    def _generate_tasks(self):
        """Generate all possible tasks from segments using adaptive sliding windows."""
        tasks = []
        
        for seg_id, seg in enumerate(self.segments):
            seg_length = seg['length']
            regime_label = seg['regime_label']
            
            # Use smaller window and stride for Regime 2
            if regime_label == 2:
                window_size = self.regime2_window_size
                stride = self.regime2_stride
                support_size = self.regime2_support_size
            else:
                window_size = self.window_size
                stride = self.stride
                support_size = self.support_size
            
            # Slide window through segment
            for start_offset in range(0, seg_length - window_size + 1, stride):
                start_idx = seg['start_idx'] + start_offset
                
                tasks.append({
                    'start_idx': start_idx,
                    'segment_id': seg_id,
                    'regime_label': regime_label,
                    'support_size': support_size,  # Store adaptive support size
                    'window_size': window_size      # Store adaptive window size
                })
        
        return tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        """
        Get a single task with adaptive window sizes.
        
        Returns:
            dict with keys:
              - support_x: torch.Tensor (10-30, N) - features (adaptive size)
              - support_y: torch.Tensor (10-30, M) - targets
              - query_x: torch.Tensor (10, N) - features (fixed size)
              - query_y: torch.Tensor (10, M) - targets
              - regime_label: int (0=bearish, 1=bullish, 2=forced selling)
              - segment_id: int
        """
        task = self.tasks[idx]
        start_idx = task['start_idx']
        support_size = task['support_size']
        window_size = task['window_size']
        
        # Extract support set (adaptive size: 30 for R0/R1, 10 for R2)
        support_data = self.features_df.iloc[start_idx:start_idx + support_size]
        support_x = torch.tensor(support_data[self.feature_cols].values, dtype=torch.float32)
        support_y = torch.tensor(support_data[self.target_cols].values, dtype=torch.float32)
        
        # Extract query set (fixed 10 days)
        query_start = start_idx + support_size
        query_data = self.features_df.iloc[query_start:query_start + self.query_size]
        query_x = torch.tensor(query_data[self.feature_cols].values, dtype=torch.float32)
        query_y = torch.tensor(query_data[self.target_cols].values, dtype=torch.float32)
        
        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y,
            'regime_label': task['regime_label'],
            'segment_id': task['segment_id']
        }
