"""
Simple MLP Portfolio Network for MAML Meta-Learning

Architecture:
- Input: (seq_len, num_features) - Variable length sequences (10-30 days)
- Hidden: 64 units with ReLU
- Output: 1 (next-day return prediction)

Designed for fast adaptation in inner loop (5-10 gradient steps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PortfolioNet(nn.Module):
    """
    Simple 2-layer MLP for next-day return prediction.
    
    Args:
        input_size: Number of features per timestep (default: 32)
        hidden_size: Hidden layer size (default: 64)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_size=32, hidden_size=64, dropout=0.1):
        super(PortfolioNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Flatten time series to vector (seq_len * features)
        # Then process through MLP
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Initialize weights (important for MAML convergence)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better MAML meta-learning"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_len, input_size) or (seq_len, input_size)
            
        Returns:
            predictions: (batch_size, 1) next-day return predictions
        """
        # Handle both batched and unbatched input
        if x.dim() == 2:  # (seq_len, features)
            x = x.unsqueeze(0)  # (1, seq_len, features)
        
        # Average pooling over time dimension (simple aggregation)
        # Alternative: use only last timestep x[:, -1, :]
        x = x.mean(dim=1)  # (batch_size, features)
        
        # MLP layers
        x = self.fc1(x)           # (batch_size, hidden_size)
        x = F.relu(x)             # ReLU activation
        x = self.dropout(x)       # Dropout for regularization
        x = self.fc2(x)           # (batch_size, 1)
        
        return x
    
    def predict_return(self, x):
        """Convenience method for single prediction (no grad)"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


if __name__ == "__main__":
    # Quick test
    print("Testing PortfolioNet...")
    
    # Create model
    model = PortfolioNet(input_size=32, hidden_size=64)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with variable-length sequences
    for seq_len in [10, 20, 30]:
        x = torch.randn(4, seq_len, 32)  # batch_size=4
        output = model(x)
        print(f"Input shape: {x.shape} -> Output shape: {output.shape}")
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    
    print("\n✅ PortfolioNet tests passed!") 
