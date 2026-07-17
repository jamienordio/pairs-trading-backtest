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

def compute_rolling_zscore(
    spread: pd.Series,
    window: int = 60,
    lag: bool = True,
) -> pd.Series:
    """
    Computes rolling z-score of spread using trailing data

    Paramters
    ---------
    spread: spread series
    window: rolling lookback window in trading days

    Returns
    -------
    zscore: rolling z-score series
    """

    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    if lag:
        rolling_mean = spread.rolling(window=window).mean().shift(1)
        rolling_std = spread.rolling(window=window).std().shift(1)
    
    zscore = (spread - rolling_mean) / rolling_std
    zscore = zscore.dropna()
    zscore.name = f"{spread.name}_rolling_zscore"
    return zscore

def compute_rolling_spread(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    window: int=252,
) -> pd.DataFrame:
    """
    Estimate rolling hedge ratios and compute rolling spread

    For each time t, alpha_t and beta_t are estimated using previous
    window observations, then applied to current prices

    Returns:
    --------
    rolling_spread: dataframe with columns: alpha, beta, spread
    """

    y = prices[ticker_y]
    x = prices[ticker_x]

    rows = []

    for i in range(window, len(prices)):
        date = prices.index[i] # get current date
        y_train = y.iloc[i - window:i] # trailing windows
        x_train = x.iloc[i - window:i]
        X_train = sm.add_constant(x_train) 
        model = sm.OLS(y_train, X_train).fit() # fit least squares
        alpha = model.params["const"]
        beta = model.params[ticker_x]
        spread_value = y.iloc[i] - alpha - beta*x.iloc[i]
        rows.append(
            {
                "date": date,
                "alpha": alpha,
                "beta": beta,
                "spread": spread_value,
            }
        )
    rolling_spread = pd.DataFrame(rows).set_index("date")
    return rolling_spread
    