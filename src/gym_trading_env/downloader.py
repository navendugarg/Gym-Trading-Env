import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import datetime
import numpy as np
import nest_asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Apply nest_asyncio if needed (e.g., for running in Jupyter)
# Consider removing if running as a standalone script without a pre-existing event loop.
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Constants ---
TIMESTAMP_OPEN_COL = "timestamp_open"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"
VOLUME_COL = "volume"
DATE_OPEN_COL = "date_open"
DATE_CLOSE_COL = "date_close"

OHLCV_COLUMNS = [TIMESTAMP_OPEN_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]
NUMERIC_COLUMNS = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]

EXCHANGE_LIMIT_RATES: Dict[str, Dict[str, int]] = {
    "bitfinex2": {
        "limit": 10_000,
        "pause_every": 1,
        "pause": 3,  # seconds
    },
    "binance": {
        "limit": 1_000,
        "pause_every": 10,
        "pause": 1,  # seconds
    },
    "huobi": {
        "limit": 1_000,
        "pause_every": 10,
        "pause": 1,  # seconds
    }
}

# --- Helper Functions ---

def _ms_to_datetime(ms: int) -> datetime.datetime:
    """Converts milliseconds timestamp to datetime object."""
    return pd.to_datetime(ms, unit="ms")

def _datetime_to_ms(dt: datetime.datetime) -> int:
    """Converts datetime object to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)

async def _fetch_ohlcv_safe(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, step_since: int) -> List[List[Any]]:
    """Safely fetches OHLCV data, handling potential exchange errors."""
    try:
        return await exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, since=step_since)
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(f"Exchange error fetching {symbol} from {exchange.id}: {e}")
        return [] # Return empty list on error

async def _ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, step_since: int, timedelta_ms: int) -> pd.DataFrame:
    """
    Fetches a single batch of OHLCV data and formats it into a DataFrame.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The trading symbol.
        timeframe: The timeframe string (e.g., '5m', '1h').
        limit: The maximum number of candles per request.
        step_since: The starting timestamp (in ms) for this batch.
        timedelta_ms: The timeframe duration in milliseconds.

    Returns:
        A pandas DataFrame with OHLCV data for the batch.
    """
    ohlcv_data = await _fetch_ohlcv_safe(exchange, symbol, timeframe, limit, step_since)
    if not ohlcv_data:
        return pd.DataFrame(columns=OHLCV_COLUMNS + [DATE_OPEN_COL, DATE_CLOSE_COL]) # Return empty DataFrame if fetch failed

    result_df = pd.DataFrame(ohlcv_data, columns=OHLCV_COLUMNS)
    result_df[NUMERIC_COLUMNS] = result_df[NUMERIC_COLUMNS].apply(pd.to_numeric)
    result_df[DATE_OPEN_COL] = result_df[TIMESTAMP_OPEN_COL].apply(_ms_to_datetime)
    result_df[DATE_CLOSE_COL] = (result_df[TIMESTAMP_OPEN_COL] + timedelta_ms).apply(_ms_to_datetime)

    return result_df

async def _download_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = '5m',
    since_ms: int = _datetime_to_ms(datetime.datetime(year=2020, month=1, day=1)),
    until_ms: int = _datetime_to_ms(datetime.datetime.now()),
    limit: int = 1000,
    pause_every: int = 10,
    pause_sec: int = 1
) -> pd.DataFrame:
    """
    Downloads historical OHLCV data for a single symbol from an exchange.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        timeframe: The timeframe string (e.g., '5m', '1h').
        since_ms: Start timestamp in milliseconds.
        until_ms: End timestamp in milliseconds.
        limit: Max candles per request (exchange-specific).
        pause_every: Number of requests before pausing.
        pause_sec: Duration of pause in seconds.

    Returns:
        A pandas DataFrame containing the downloaded OHLCV data, indexed by date_open.
    """
    timedelta_ms = int(pd.Timedelta(timeframe).total_seconds() * 1000)
    if timedelta_ms == 0:
         raise ValueError(f"Invalid timeframe: {timeframe}")

    tasks = []
    results_list: List[pd.DataFrame] = []
    current_step_ms = since_ms

    logger.info(f"Starting download for {symbol} on {exchange.id} from {pd.to_datetime(since_ms, unit='ms')} to {pd.to_datetime(until_ms, unit='ms')}")

    while current_step_ms < until_ms:
        tasks.append(
            asyncio.create_task(_ohlcv(exchange, symbol, timeframe, limit, current_step_ms, timedelta_ms))
        )
        current_step_ms += limit * timedelta_ms

        if len(tasks) >= pause_every:
            batch_results = await asyncio.gather(*tasks)
            results_list.extend(batch_results)
            logger.debug(f"Fetched batch for {symbol}, pausing for {pause_sec}s...")
            await asyncio.sleep(pause_sec)
            tasks = []

    # Process any remaining tasks
    if tasks:
        batch_results = await asyncio.gather(*tasks)
        results_list.extend(batch_results)

    if not results_list:
        logger.warning(f"No data downloaded for {symbol} on {exchange.id}.")
        return pd.DataFrame()

    # Combine, filter, and clean the data
    final_df = pd.concat(results_list, ignore_index=True)
    final_df = final_df.loc[
        (final_df[TIMESTAMP_OPEN_COL] >= since_ms) & (final_df[TIMESTAMP_OPEN_COL] < until_ms)
    ]
    final_df.drop(columns=[TIMESTAMP_OPEN_COL], inplace=True) # Drop raw timestamp
    final_df.set_index(DATE_OPEN_COL, drop=True, inplace=True)
    final_df.sort_index(inplace=True)
    final_df.dropna(inplace=True)
    # Use keep='first' to maintain chronological order if duplicates exist
    final_df = final_df[~final_df.index.duplicated(keep='first')]

    logger.info(f"Finished download for {symbol} on {exchange.id}. Rows: {len(final_df)}")
    return final_df

async def _download_symbols(
    exchange_name: str,
    symbols: List[str],
    data_dir: Path,
    timeframe: str,
    **kwargs: Any
) -> None:
    """
    Downloads data for multiple symbols from a single exchange and saves them to pickle files.

    Args:
        exchange_name: Name of the exchange (e.g., 'binance').
        symbols: List of symbols to download.
        data_dir: Directory Path object to save the data files.
        timeframe: The timeframe string.
        **kwargs: Additional arguments passed to _download_symbol (since_ms, until_ms, etc.).
    """
    exchange = None # Initialize exchange to None
    try:
        exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        data_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        for symbol in symbols:
            logger.info(f"Requesting download for {symbol} from {exchange_name}...")
            df = await _download_symbol(exchange=exchange, symbol=symbol, timeframe=timeframe, **kwargs)

            if not df.empty:
                # Sanitize symbol for filename
                safe_symbol = symbol.replace('/', '_')
                save_file = data_dir / f"{exchange_name}-{safe_symbol}-{timeframe}.pkl"
                try:
                    df.to_pickle(save_file)
                    logger.info(f"{symbol} data from {exchange_name} saved to {save_file}")
                except IOError as e:
                    logger.error(f"Failed to save data for {symbol} to {save_file}: {e}")
            else:
                 logger.warning(f"No data returned for {symbol} from {exchange_name}, skipping save.")

    except (ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, AttributeError) as e:
         logger.error(f"Failed to initialize or connect to exchange {exchange_name}: {e}")
    except Exception as e: # Catch other potential errors during download
         logger.error(f"An unexpected error occurred during download for {exchange_name}: {e}")
    finally:
        if exchange:
            await exchange.close()
            logger.debug(f"Closed connection to {exchange_name}")


async def _download(
    exchange_names: List[str],
    symbols: List[str],
    timeframe: str,
    data_dir: str, # Keep as string for input, convert to Path internally
    since: datetime.datetime,
    until: Optional[datetime.datetime] = None
) -> None:
    """
    Coordinates the download of OHLCV data from multiple exchanges for multiple symbols.

    Args:
        exchange_names: List of exchange names.
        symbols: List of symbols.
        timeframe: Timeframe string.
        data_dir: Directory path string where data will be saved.
        since: Start datetime.
        until: End datetime (defaults to now if None).
    """
    if until is None:
        until = datetime.datetime.now(datetime.timezone.utc) # Use timezone-aware now

    # Ensure 'since' is also timezone-aware (assuming UTC if naive)
    if since.tzinfo is None:
        since = since.replace(tzinfo=datetime.timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=datetime.timezone.utc)


    since_ms = _datetime_to_ms(since)
    until_ms = _datetime_to_ms(until)
    target_dir = Path(data_dir) # Convert string path to Path object

    tasks = []
    for exchange_name in exchange_names:
        if exchange_name not in EXCHANGE_LIMIT_RATES:
            logger.warning(f"Rate limit configuration not found for {exchange_name}. Using default values.")
            # Provide some defaults or skip
            limit = 1000
            pause_every = 10
            pause_sec = 1
        else:
            config = EXCHANGE_LIMIT_RATES[exchange_name]
            limit = config["limit"]
            pause_every = config["pause_every"]
            pause_sec = config["pause"]

        tasks.append(
            _download_symbols(
                exchange_name=exchange_name,
                symbols=symbols,
                timeframe=timeframe,
                data_dir=target_dir,
                limit=limit,
                pause_every=pause_every,
                pause_sec=pause_sec,
                since_ms=since_ms,
                until_ms=until_ms
            )
        )
    await asyncio.gather(*tasks)
    logger.info("All download tasks completed.")

def download(
    exchange_names: List[str],
    symbols: List[str],
    timeframe: str,
    data_dir: str,
    since: datetime.datetime,
    until: Optional[datetime.datetime] = None
) -> None:
    """
    Public function to trigger the asynchronous download process.

    Args:
        exchange_names: List of exchange names.
        symbols: List of symbols.
        timeframe: Timeframe string.
        data_dir: Directory path string where data will be saved.
        since: Start datetime.
        until: End datetime (defaults to now if None).
    """
    logger.info("Initiating data download...")
    try:
        asyncio.run(
            _download(exchange_names, symbols, timeframe, data_dir, since, until)
        )
        logger.info("Data download process finished.")
    except Exception as e:
        logger.exception(f"An error occurred during the download process: {e}")


async def main() -> None:
    """Example usage function."""
    await _download(
        ["binance", "bitfinex2", "huobi"],
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="30m",
        data_dir="test/data", # Relative path example
        since=datetime.datetime(year=2023, month=1, day=1), # Shorter period for testing
        until=datetime.datetime(year=2023, month=1, day=2)
    )

if __name__ == "__main__":
    # Example of calling the public download function
    download(
        exchange_names=["binance"], # Example: download only from binance
        symbols=["BTC/USDT"],       # Example: download only BTC/USDT
        timeframe="1h",
        data_dir="downloaded_data", # Example: different directory
        since=datetime.datetime(2023, 10, 1),
        until=datetime.datetime(2023, 10, 5)
    )

    # Or run the original main example function
    # logger.info("Running main example function...")
    # asyncio.run(main())
