import numpy as np
import pandas as pd

def compute_cumulative_returns(
    returns: pd.Series,
) -> pd.Series:
    """
    Compute cumulative returns from daily strategy returns
    """
    cumulative = (1+returns).cumprod()
    cumulative.name = f"{returns.name}_cumulative_returns"
    return cumulative

def compute_performance_metrics(
    returns: pd.Series,
    periods_per_year: int=252,
) -> dict:
    """
    Compute basic performance metrics for a daily return series

    Metrics:
    --------
    total_return
    annualized_return
    annualized_volatility
    sharpe_ratio
    max_drawdown
    """
    cumulative = compute_cumulative_returns(returns)
    
    total_return = cumulative.iloc[-1] - 1
    
    n_periods = len(returns)
    annualized_return = cumulative.iloc[-1]**(periods_per_year/n_periods) - 1
    
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)

    if returns.std() == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    running_max = cumulative.cummax()
    drawdown = cumulative/running_max - 1
    max_drawdown = drawdown.min()
    
    metrics = {
    "total_return": total_return,
    "annualized_return": annualized_return,
    "annualized_volatility": annualized_volatility,
    "sharpe_ratio": sharpe_ratio,
    "max_drawdown": max_drawdown,
    }
    return metrics