import pandas as pd

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns from price data

    Parameters
    ----------
    prices: DataFrame of prices with dates as index and tickers as columns

    Returns
    -------
    returns: DataFrame of daily simple returns
    """

    returns = prices.pct_change()
    returns = returns.dropna()

    return returns