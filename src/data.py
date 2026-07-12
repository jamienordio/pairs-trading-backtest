from pathlib import Path
import time

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw"


def download_prices(
    tickers: list[str] | str,
    start: str,
    end: str,
    cache_dir: str | Path | None = None,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for one or more tickers.

    Cached files are stored in the project-level data/raw folder by default.
    """

    if isinstance(tickers, str):
        tickers = [tickers]

    if cache_dir is None:
        cache_path = DEFAULT_CACHE_DIR
    else:
        cache_path = Path(cache_dir)

    cache_path.mkdir(parents=True, exist_ok=True)

    price_series = []

    for ticker in tickers:
        file_path = cache_path / f"{ticker}_{start}_{end}.csv"

        if file_path.exists():
            print(f"Loading {ticker} from cache...")
            close = pd.read_csv(file_path, index_col=0, parse_dates=True)[ticker]
            price_series.append(close)
            continue

        print(f"Downloading {ticker}...")

        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        close = data["Close"].copy()
        close.name = ticker

        close.to_csv(file_path)
        price_series.append(close)

        time.sleep(sleep_seconds)

    prices = pd.concat(price_series, axis=1)
    prices = prices.dropna()
    prices = prices.sort_index()
    prices.columns.name = None

    if prices.empty:
        raise ValueError("Price DataFrame is empty after cleaning.")

    return prices