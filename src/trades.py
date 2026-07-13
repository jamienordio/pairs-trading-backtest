import pandas as pd

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