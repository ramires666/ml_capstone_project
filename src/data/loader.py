"""
Data Loader Module for Bitcoin Price Prediction.

This module handles loading and merging data from various sources:
- Klines (OHLCV candlestick data)
- Funding rates (futures funding rate)
- Volume breakdown (buy/sell volumes from aggregated trades)

The main function `load_and_merge_data()` combines all sources into a single
DataFrame ready for feature engineering.

Data Flow:
----------
1. Load klines.parquet     -> Basic OHLCV data
2. Load fundings.parquet   -> Funding rates (filled forward)
3. Load volumes.parquet    -> Buy/sell volume breakdown
4. Merge all on 'time' column
5. Sort by time, handle missing values
6. Optionally filter by date range

Author: Capstone Project
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Default Data Paths
# =============================================================================
# These paths assume data is stored in the project's data/processed/ directory
# Parser saves files there after running: python -m src.data.parser

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
DEFAULT_KLINES_PATH = DATA_DIR / "klines_15min_all.parquet"
DEFAULT_FUNDINGS_PATH = DATA_DIR / "fundings.parquet"
DEFAULT_VOLUMES_PATH = DATA_DIR / "aggtrades_15min_all.parquet"


def load_klines(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load klines (candlestick) data from Parquet file.
    
    Klines contain the core OHLCV data:
    - open_time: Candle start timestamp
    - open, high, low, close: Price data
    - volume: Trading volume in base asset
    - quote_volume: Trading volume in quote asset (USDT)
    
    Parameters
    ----------
    filepath : str, optional
        Path to klines Parquet file. Uses default if not specified.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, open, high, low, close, volume, etc.
    """
    path = Path(filepath) if filepath else DEFAULT_KLINES_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Klines file not found: {path}")
    
    df = pd.read_parquet(path)
    
    # Standardize time column name
    if 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'time'})
    
    # Ensure time is datetime with UTC timezone
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    return df


def load_funding_rates(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load funding rates from Parquet file.
    
    Funding rates are periodic payments between long/short positions
    in perpetual futures. They indicate market sentiment:
    - Positive funding = longs pay shorts (bullish sentiment)
    - Negative funding = shorts pay longs (bearish sentiment)
    
    Parameters
    ----------
    filepath : str, optional
        Path to fundings Parquet file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, funding_rate
    """
    path = Path(filepath) if filepath else DEFAULT_FUNDINGS_PATH
    
    if not path.exists():
        print(f"⚠️ Funding rates file not found: {path}")
        return pd.DataFrame()  # Return empty DataFrame
    
    df = pd.read_parquet(path)
    
    # Standardize time column name
    if 'calc_time' in df.columns:
        df = df.rename(columns={'calc_time': 'time'})
    elif 'fundingTime' in df.columns:
        df = df.rename(columns={'fundingTime': 'time'})
    
    # Ensure time is datetime with UTC timezone
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    # Keep only necessary columns
    if 'funding_rate' in df.columns:
        df = df[['time', 'funding_rate']]
    elif 'fundingRate' in df.columns:
        df = df.rename(columns={'fundingRate': 'funding_rate'})
        df = df[['time', 'funding_rate']]
    
    return df


def load_volumes(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load volume breakdown from Parquet file.
    
    Volume breakdown shows buy vs sell volume, which helps identify:
    - Buying pressure (more buy volume = bullish)
    - Selling pressure (more sell volume = bearish)
    
    NOTE: Only loads volume-related columns (buy_vol, sell_vol, total_vol).
    OHLC columns are excluded to prevent conflicts with klines data.
    
    Parameters
    ----------
    filepath : str, optional
        Path to volumes Parquet file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, buy_vol, sell_vol, total_vol
    """
    path = Path(filepath) if filepath else DEFAULT_VOLUMES_PATH
    
    if not path.exists():
        print(f"⚠️ Volumes file not found: {path}")
        return pd.DataFrame()  # Return empty DataFrame
    
    df = pd.read_parquet(path)
    
    # Standardize time column name
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'time'})
    
    # Ensure time is datetime with UTC timezone
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    # Only keep volume columns (exclude OHLC to prevent conflicts with klines)
    volume_cols = ['time', 'buy_vol', 'sell_vol', 'total_vol']
    available_cols = [c for c in volume_cols if c in df.columns]
    df = df[available_cols].copy()
    
    return df


def load_and_merge_data(klines_path: Optional[str] = None,
                        fundings_path: Optional[str] = None,
                        volumes_path: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        use_merged: bool = True) -> pd.DataFrame:
    """
    Load and merge all data sources into a single DataFrame.
    
    This is the main data loading function. It:
    1. Checks for pre-merged all_merged.parquet (fast path)
    2. If not found, loads and merges individual files
    3. Filters by date range if specified
    4. Creates log returns
    
    Parameters
    ----------
    klines_path : str, optional
        Path to klines Parquet file.
    fundings_path : str, optional
        Path to funding rates Parquet file.
    volumes_path : str, optional
        Path to volumes Parquet file.
    start_date : str, optional
        Filter data from this date (format: '2024-01-01').
    end_date : str, optional
        Filter data until this date.
    use_merged : bool, default=True
        If True, use all_merged.parquet if it exists (faster).
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all data sources.
    
    Examples
    --------
    >>> df = load_and_merge_data()
    >>> print(df.columns.tolist())
    ['time', 'open', 'high', 'low', 'close', 'volume', 
     'funding_rate', 'bid_vol', 'ask_vol', 'delta_vol', 'cvd', 'log_return']
    """
    print("\n" + "=" * 60)
    print("📥 LOADING DATA")
    print("=" * 60)
    
    # Fast path: use pre-merged file if it exists
    merged_path = DATA_DIR / "all_merged.parquet"
    if use_merged and merged_path.exists():
        print(f"\n✅ Loading pre-merged dataset: {merged_path}")
        df = pd.read_parquet(merged_path)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        
        # Filter by date range
        if start_date:
            start_dt = pd.to_datetime(start_date, utc=True)
            df = df[df['time'] >= start_dt]
            print(f"⏰ Filtered from: {start_date}")
        
        if end_date:
            end_dt = pd.to_datetime(end_date, utc=True)
            df = df[df['time'] <= end_dt]
            print(f"⏰ Filtered until: {end_date}")
        
        # Add log returns if missing
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        df = df.reset_index(drop=True)
        
        print(f"\n📊 Data Summary:")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
        
        return df
    
    # Slow path: merge individual files
    print("\n⚠️ Pre-merged file not found, merging individual files...")
    print("   TIP: Run 'python -m src.data.parser' to create all_merged.parquet")
    
    # Step 1: Load klines (required)
    print("\n1️⃣ Loading klines...")
    df = load_klines(klines_path)
    print(f"   Loaded {len(df):,} rows")
    
    # Step 2: Load and merge funding rates (optional)
    print("\n2️⃣ Loading funding rates...")
    fundings = load_funding_rates(fundings_path)
    
    if not fundings.empty:
        # Left join: keep all klines, add funding where available
        df = pd.merge(df, fundings, on='time', how='left')
        
        # Forward fill funding rates (they update every 8 hours typically)
        # This fills gaps between funding rate updates
        if 'funding_rate' in df.columns:
            df['funding_rate'] = df['funding_rate'].ffill()
        print(f"   Merged funding rates")
    else:
        print(f"   Skipped (file not found)")
    
    # Step 3: Load and merge volume breakdown (optional)
    print("\n3️⃣ Loading volume breakdown...")
    volumes = load_volumes(volumes_path)
    
    if not volumes.empty:
        df = pd.merge(df, volumes, on='time', how='left')
        print(f"   Merged volume breakdown")
    else:
        print(f"   Skipped (file not found)")
    
    # Step 4: Sort by time and reset index
    df = df.sort_values('time').reset_index(drop=True)
    
    # Step 5: Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        df = df[df['time'] >= start_dt]
        print(f"\n⏰ Filtered from: {start_date}")
    
    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True)
        df = df[df['time'] <= end_dt]
        print(f"⏰ Filtered until: {end_date}")
    
    # Step 6: Create log returns
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Summary
    print(f"\n📊 Data Summary:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"   Columns: {df.columns.tolist()}")
    
    return df


def split_data_by_time(df: pd.DataFrame,
                       train_end: str,
                       test_start: str,
                       val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets by time.
    
    Time-based splitting is CRUCIAL for financial data to avoid look-ahead bias.
    Never use random splitting for time series!
    
    Split Structure:
    ----------------
    Train: data before train_end
    Val:   last val_ratio of training data (carved out)
    Test:  data from test_start onwards
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'time' column.
    train_end : str
        End date for training data (format: '2024-06-30').
    test_start : str
        Start date for test data (format: '2024-07-01').
    val_ratio : float, default=0.1
        Fraction of training data to use for validation.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df
    
    Examples
    --------
    >>> train, val, test = split_data_by_time(
    ...     df, train_end='2024-06-30', test_start='2024-07-01'
    ... )
    """
    # Convert dates to datetime with UTC
    train_end_dt = pd.to_datetime(train_end, utc=True)
    test_start_dt = pd.to_datetime(test_start, utc=True)
    
    # Split by time
    train_val_df = df[df['time'] <= train_end_dt].copy()
    test_df = df[df['time'] >= test_start_dt].copy()
    
    # Split train into train and validation (last val_ratio of train)
    val_size = int(len(train_val_df) * val_ratio)
    train_df = train_val_df.iloc[:-val_size].copy()
    val_df = train_val_df.iloc[-val_size:].copy()
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(train_df):,} rows ({train_df['time'].min()} to {train_df['time'].max()})")
    print(f"   Val:   {len(val_df):,} rows ({val_df['time'].min()} to {val_df['time'].max()})")
    print(f"   Test:  {len(test_df):,} rows ({test_df['time'].min()} to {test_df['time'].max()})")
    
    return train_df, val_df, test_df


# =============================================================================
# Module Test
# =============================================================================
if __name__ == "__main__":
    """Test data loading with available files."""
    
    print("=" * 60)
    print("Testing Data Loader Module")
    print("=" * 60)
    
    try:
        df = load_and_merge_data()
        print("\n✅ Data loaded successfully!")
        print(f"\nFirst 5 rows:")
        print(df.head())
    except FileNotFoundError as e:
        print(f"\n⚠️ Test skipped - data files not found. Run downloader first.")
        print(f"   Error: {e}")
