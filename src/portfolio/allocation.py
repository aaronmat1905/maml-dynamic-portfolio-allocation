"""
Portfolio allocation strategies based on MAML predictions and jump regimes.

This module converts MAML return predictions into portfolio weights (equity/cash)
using regime-aware allocation rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def prediction_to_weights(
    prediction: float,
    regime: int,
    base_allocations: Optional[Dict[int, float]] = None,
    sensitivity: float = 2.0,
    min_equity: float = 0.0,
    max_equity: float = 1.0
) -> Dict[str, float]:
    """
    Convert MAML return prediction and jump regime to portfolio weights.
    
    Strategy:
    - Base allocation depends on regime (higher risk = lower equity)
    - Adjust allocation based on prediction magnitude and sign
    - Positive prediction → increase equity, negative → decrease equity
    
    Args:
        prediction: MAML predicted next-day return (e.g., 0.02 for +2%)
        regime: Jump regime label (0=calm JR0, 1=moderate JR1, 2=crisis JR2)
        base_allocations: Base equity allocation per regime. Default:
            {0: 0.60, 1: 0.50, 2: 0.30}  # JR0=60%, JR1=50%, JR2=30%
        sensitivity: How much to adjust allocation per 1% predicted return.
            Default 2.0 means +1% prediction → +2% equity weight
        min_equity: Minimum equity allocation (default 0%)
        max_equity: Maximum equity allocation (default 100%)
    
    Returns:
        Dictionary with 'equity' and 'cash' weights (sum to 1.0)
    
    Example:
        >>> # JR2 crisis, predict +2% return
        >>> weights = prediction_to_weights(0.02, regime=2)
        >>> # Base 30% + adjustment → ~34% equity
        
        >>> # JR0 calm, predict -1% return
        >>> weights = prediction_to_weights(-0.01, regime=0)
        >>> # Base 60% - adjustment → ~58% equity
    """
    if base_allocations is None:
        # Conservative defaults: reduce equity in higher risk regimes
        base_allocations = {
            0: 0.60,  # JR0 (calm): 60% equity
            1: 0.50,  # JR1 (moderate): 50% equity
            2: 0.30,  # JR2 (crisis): 30% equity
        }
    
    # Get base allocation for this regime
    base_equity = base_allocations.get(regime, 0.50)
    
    # Calculate adjustment based on prediction
    # prediction is in decimal (e.g., 0.02 for 2% return)
    # sensitivity=2.0 means 1% prediction → 2% weight adjustment
    adjustment = prediction * 100 * sensitivity / 100  # Convert to weight change
    
    # Apply adjustment to base allocation
    equity_weight = base_equity + adjustment
    
    # Clip to valid range
    equity_weight = np.clip(equity_weight, min_equity, max_equity)
    
    # Cash is the remainder
    cash_weight = 1.0 - equity_weight
    
    return {
        'equity': float(equity_weight),
        'cash': float(cash_weight)
    }


def compute_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Compute portfolio performance metrics.
    
    Args:
        returns: Series of portfolio returns (daily)
        benchmark_returns: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate for Sharpe ratio (default 2%)
    
    Returns:
        Dictionary with performance metrics:
        - total_return: Cumulative return over period
        - annualized_return: Annualized return (assuming 252 trading days)
        - annualized_volatility: Annualized volatility
        - sharpe_ratio: (Return - RFR) / Volatility
        - max_drawdown: Maximum peak-to-trough decline
        - calmar_ratio: Return / Max Drawdown
        - win_rate: Percentage of positive return days
        - avg_win: Average return on positive days
        - avg_loss: Average return on negative days
        - If benchmark provided:
          - alpha: Excess return vs benchmark
          - beta: Correlation with benchmark
          - tracking_error: Volatility of excess returns
    """
    if len(returns) == 0:
        raise ValueError("Returns series is empty")
    
    # Remove any NaN values
    returns = returns.dropna()
    
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    n_days = len(returns)
    n_years = n_days / 252  # Assume 252 trading days per year
    
    if n_years > 0:
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annualized_return = 0.0
    
    # Volatility
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    if annualized_volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    else:
        sharpe_ratio = 0.0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    if max_drawdown < 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = 0.0
    
    # Win/loss statistics
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    
    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
    }
    
    # Benchmark comparison if provided
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        # Align returns and benchmark
        aligned = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) > 1:
            # Alpha: excess return
            benchmark_total = (1 + aligned['benchmark']).prod() - 1
            benchmark_ann = (1 + benchmark_total) ** (1 / n_years) - 1 if n_years > 0 else 0.0
            alpha = annualized_return - benchmark_ann
            
            # Beta: regression coefficient
            covariance = aligned['portfolio'].cov(aligned['benchmark'])
            benchmark_variance = aligned['benchmark'].var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
            
            # Tracking error
            excess_returns = aligned['portfolio'] - aligned['benchmark']
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            metrics['alpha'] = float(alpha)
            metrics['beta'] = float(beta)
            metrics['tracking_error'] = float(tracking_error)
    
    return metrics


def backtest_strategy(
    predictions: pd.Series,
    regimes: pd.Series,
    actual_returns: pd.Series,
    strategy_name: str = "MAML",
    **allocation_kwargs
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Backtest a portfolio strategy using MAML predictions and regimes.
    
    Args:
        predictions: Series of MAML predictions (index=dates, values=predicted returns)
        regimes: Series of regime labels (index=dates, values=0/1/2)
        actual_returns: Series of actual market returns (index=dates)
        strategy_name: Name for this strategy (for logging)
        **allocation_kwargs: Additional arguments for prediction_to_weights()
    
    Returns:
        Tuple of (portfolio_returns, metrics)
        - portfolio_returns: Series of realized portfolio returns
        - metrics: Dictionary from compute_portfolio_metrics()
    
    Example:
        >>> returns, metrics = backtest_strategy(
        ...     predictions=maml_preds,
        ...     regimes=jump_regimes,
        ...     actual_returns=sp500_returns,
        ...     base_allocations={0: 0.60, 1: 0.50, 2: 0.30}
        ... )
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    # Align all series
    data = pd.DataFrame({
        'prediction': predictions,
        'regime': regimes,
        'market_return': actual_returns
    }).dropna()
    
    if len(data) == 0:
        raise ValueError("No overlapping data between predictions, regimes, and returns")
    
    # Calculate portfolio returns for each day
    portfolio_returns = []
    
    for idx, row in data.iterrows():
        # Get allocation weights based on prediction and regime
        weights = prediction_to_weights(
            prediction=row['prediction'],
            regime=int(row['regime']),
            **allocation_kwargs
        )
        
        # Portfolio return = equity_weight * market_return + cash_weight * 0
        # (assuming cash earns 0% daily return)
        portfolio_return = weights['equity'] * row['market_return']
        portfolio_returns.append(portfolio_return)
    
    portfolio_returns = pd.Series(portfolio_returns, index=data.index, name=strategy_name)
    
    # Compute metrics
    metrics = compute_portfolio_metrics(
        portfolio_returns,
        benchmark_returns=data['market_return']
    )
    
    return portfolio_returns, metrics


def create_baseline_strategies(actual_returns: pd.Series) -> Dict[str, pd.Series]:
    """
    Create baseline strategy returns for comparison.
    
    Args:
        actual_returns: Series of market returns
    
    Returns:
        Dictionary mapping strategy name to returns series:
        - 'Buy-and-Hold': 100% equity always
        - '60/40': 60% equity, 40% cash always
        - '40/60': 40% equity, 60% cash always
    """
    baselines = {}
    
    # Buy and hold (100% equity)
    baselines['Buy-and-Hold'] = actual_returns.copy()
    
    # 60/40 portfolio
    baselines['60/40'] = actual_returns * 0.6
    
    # 40/60 portfolio (conservative)
    baselines['40/60'] = actual_returns * 0.4
    
    return baselines
