import pandas as pd
from itertools import combinations

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
    