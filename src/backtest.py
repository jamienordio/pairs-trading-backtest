import pandas as pd

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