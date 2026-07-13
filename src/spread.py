import pandas as pd
import statsmodels.api as sm

def estimate_hedge_ratio(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> tuple[float, float]:
    """
    Estimate the hedge ratio beta from the regression 
    Y_t = alpha + beta X_t + residual_t

    Parameters:
    -----------
    prices: DataFrame of adjusted close prices
    ticker_y: dependent ticker
    ticker_x: independent ticker

    Returns:
    --------
    alpha: regression intercept
    beta: hedge ratio
    """
    y = prices[ticker_y]
    x = prices[ticker_x]

    X = sm.add_constant(x)

    model = sm.OLS(y,X).fit()

    alpha = model.params["const"]
    beta = model.params[ticker_x]
    
    return [alpha, beta]

def compute_spread(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    alpha: float,
    beta: float,
) -> pd.Series:
    """
    Compute the hedged spread
    """
    spread = prices[ticker_y] - alpha - beta*prices[ticker_x]
    spread.name = f"{ticker_y}_{ticker_x}_spread"
    return spread

def compute_zscore(spread: pd.Series) -> pd.Series:
    """
    Compute the full-sample z-score of a spread

    z_t = (spread_t - mean(spread)) / str(spread)

    Note:
    -----
    exploration only, not to be used for real backtest.
    """
    mean = spread.mean()
    std = spread.std()

    zscore = (spread - mean) / std

    zscore.name = f"{spread.name}_zscore"
    
    return zscore