import pandas as pd
from itertools import combinations
import statsmodels.api as sm
import numpy as np

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
    # computes fractional change between current and prev element
    returns = prices.pct_change()
    returns = returns.dropna()

    return returns

def compute_return_correlations(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of daily returns

    Parameters
    ----------
    returns: dataframe of daily simple returns

    Returns
    -------
    corr: correlation matrix of returns
    """
    return returns.corr()

def get_unique_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    """
    Create all unqiue ticker pairs
    """
    return list(combinations(tickers, 2))

def rank_pairs_by_correlation(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank all unique ticker pairs by daily return correlation

    Parameters
    ----------
    returns: DataFrame of daily simple returns

    Returns
    -------
    ranked_pairs: DataFrame with columns: ticker_1, ticker_2, correlation
    """
    corr = compute_return_correlations(returns)
    tickers = list(returns.columns)
    pairs = get_unique_pairs(tickers)

    rows = []

    for ticker_1, ticker_2 in pairs:
        correlation = corr[ticker_1][ticker_2]
        rows.append({
            "ticker_1": ticker_1,
            "ticker_2": ticker_2,
            "correlation": correlation,
        })
        pass

    ranked_pairs = pd.DataFrame(rows)

    ranked_pairs = ranked_pairs.sort_values(
        by="correlation",
        ascending=False,
    )

    return ranked_pairs

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
        rolling_mean =  spread.rolling(window=window).mean().shift(1)
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

def generate_spread_positions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """
    Generate spread position from z-scores.

    Position convention:
     1 = long spread
    -1 = short spread
     0 = flat
    """

    positions = []
    current_position = 0

    for z in zscore:
        if current_position == 0:
            if z < -entry_threshold:
                current_position = 1
            elif z > entry_threshold:
                current_position = -1

        else:
            if abs(z) < exit_threshold:
                current_position = 0

        positions.append(current_position)

    positions = pd.Series(
        positions,
        index=zscore.index,
        name=f"{zscore.name}_position",
    )

    return positions

def compute_strategy_returns(
    prices: pd.DataFrame,
    positions: pd.Series,
    ticker_y: str,
    ticker_x: str,
    beta: float,
    transaction_cost_bps: float=0.0,
) -> pd.Series:
    """
    Compute daily returns for a pairs-trading spread strategy

    Spread definition:
    ------------------
    spread_t = Y_t - alpha - beta X_t

    Position convention:
    --------------------
     1 = long spread
    -1 = short spread
     0 = flat

     Returns:
     --------
     strategy_returns: daily strategy returns
    """
    y = prices[ticker_y]
    x = prices[ticker_x]

    long_spread_pnl = y.diff() - beta * x.diff()

    gross_exposure = y.shift(1).abs() + abs(beta) * x.shift(1).abs()

    lagged_positions = positions.shift(1).fillna(0)

    strategy_returns = lagged_positions * long_spread_pnl / gross_exposure

    cost_per_turnover = transaction_cost_bps / 10_000
    turnover = positions.diff().abs().shift(1).fillna(0)
    transaction_costs = cost_per_turnover * turnover

    strategy_returns = strategy_returns - transaction_costs

    strategy_returns = strategy_returns.dropna()
    strategy_returns.name = f"{ticker_y}_{ticker_x}_strategy_returns"

    return strategy_returns

def compute_strategy_returns_dynamic_beta(
    prices: pd.DataFrame,
    positions: pd.Series,
    beta: pd.Series,
    ticker_y: str,
    ticker_x: str,
    transaction_cost_bps: float = 0.0,
) -> pd.Series:
    """
    Compute daily strategy returns using time-varying hedge ratio beta_t

    Positions
    ---------
     1 = long spread
    -1 = short spread
     0 = flat
    """
    y = prices[ticker_y]
    x = prices[ticker_x]
    
    beta_lagged = beta.shift(1) # previous deta beta
    long_spread_pnl = y.diff() - beta_lagged*x.diff() # long-spread P&L
    gross_exposure = y.shift(1).abs() + beta_lagged.abs()*x.shift(1).abs()
    lagged_positions = positions.shift(1).fillna(0) # lag positions by 1 day
    turnover = positions.diff().abs().shift(1).fillna(0)

    df = pd.concat(
        {
            "position": lagged_positions,
            "long_spread_pnl": long_spread_pnl,
            "gross_exposure": gross_exposure,
            "turnover": turnover,
        },
        axis=1,
    ).dropna()

    cost_per_turnover = transaction_cost_bps / 10_000

    strategy_returns = (
        df["position"] * df["long_spread_pnl"] / df["gross_exposure"]
        - cost_per_turnover * df["turnover"]
    )

    strategy_returns.name = f"{ticker_y}_{ticker_x}_dynamic_beta_strategy_returns"
    return strategy_returns

def run_dynamic_beta_backtest(
    prices,
    ticker_y,
    ticker_x,
    hedge_window=252,
    zscore_window=60,
    entry_threshold=2.0,
    exit_threshold=0.5,
    transaction_cost_bps=5.0,
    evaluation_start=None,
    evaluation_end=None,
):
    rolling_spread_df = compute_rolling_spread(
        prices=prices,
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        window=hedge_window,
    )

    rolling_spread = rolling_spread_df["spread"]
    rolling_beta = rolling_spread_df["beta"]

    rolling_zscore = compute_rolling_zscore(
        spread=rolling_spread,
        window=zscore_window,
        lag=True,
    )

    positions = generate_spread_positions(
        zscore=rolling_zscore,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )

    strategy_returns = compute_strategy_returns_dynamic_beta(
        prices=prices,
        positions=positions,
        beta=rolling_beta,
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        transaction_cost_bps=transaction_cost_bps,
    )

    if evaluation_start is not None:
        strategy_returns = strategy_returns.loc[evaluation_start:]

    if evaluation_end is not None:
        strategy_returns = strategy_returns.loc[:evaluation_end]

    positions_eval = positions.reindex(strategy_returns.index).ffill().fillna(0)

    metrics = compute_performance_metrics(strategy_returns)

    trade_log = build_trade_log(
        positions=positions,
        strategy_returns=strategy_returns,
    )

    metrics["n_trades"] = len(trade_log)
    metrics["win_rate"] = (
        (trade_log["trade_return"] > 0).mean()
        if len(trade_log) > 0
        else float("nan")
    )
    metrics["avg_holding_days"] = (
        trade_log["holding_days"].mean()
        if len(trade_log) > 0
        else float("nan")
    )
    metrics["days_in_market_frac"] = (positions_eval != 0).mean()

    return metrics

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

def build_trade_log(
    positions: pd.Series,
    strategy_returns: pd.Series,
) -> pd.DataFrame:
    """
    Build a trade-level summary from position and strategy return series

    Parameters:
    -----------
    positions: Series of positions: 1 long spread, -1 short spread, 0 flat

    strategy_returns: daily strategy returns, including transaction costs

    Returns:
    --------
    trade_log: DataFrame with one row per completed trade
    """
    trades = []

    current_position = 0
    entry_date = None

    for date, position in positions.items():
        # entry
        if current_position == 0 and position != 0:
            current_position = position
            entry_date = date

        # exit
        elif current_position != 0 and position == 0:
            exit_date = date

            trade_returns = strategy_returns.loc[entry_date:exit_date]
            trade_return = (1 + trade_returns).prod() - 1

            direction = "long_spread" if current_position == 1 else "short_spread"
            holding_days = len(positions.loc[entry_date:exit_date])
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "direction": direction,
                    "holding_days": holding_days,
                    "trade_return": trade_return,
                }
            )

            current_position = 0
            entry_date = None
            
    trade_log = pd.DataFrame(trades)

    if not trade_log.empty:
        trade_log = trade_log[
        ["entry_date", "exit_date", "direction", "holding_days", "trade_return"]
        ]
    return trade_log
