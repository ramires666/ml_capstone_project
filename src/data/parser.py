"""
Data Parser for Binance Archives

Parses downloaded ZIP archives into Parquet files.
Based on old_project_files/data/parsing_klines.py with improvements.
"""

import sys
from pathlib import Path
import zipfile
import pandas as pd
from typing import Optional, List
from tqdm import tqdm


def read_klines_zip(path: Path) -> pd.DataFrame:
    """
    Read a single Binance klines archive (ZIP with CSV inside).
    
    Handles both formats:
    - CSV without headers (older Binance archives)
    - CSV with headers (newer Binance archives)
    
    Expected columns:
    open_time, open, high, low, close, volume, close_time, quote_volume, count,
    taker_buy_volume, taker_buy_quote_volume, ignore
    
    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume, 
        quote_volume, taker_buy_volume, taker_buy_quote_volume, count
    """
    path = Path(path)
    
    # Standard Binance klines column names
    expected_cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ]
    
    with zipfile.ZipFile(path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV entries found inside {path}")
        
        member = csv_members[0]
        
        # First, peek at the first row to detect if there's a header
        with zf.open(member) as f:
            first_line = f.readline().decode('utf-8').strip()
        
        # Check if first line contains text headers (not numeric timestamps)
        has_header = 'open_time' in first_line.lower() or not first_line.split(',')[0].isdigit()
        
        # Re-read the file with correct header setting
        with zf.open(member) as f:
            if has_header:
                # File has headers - read them directly
                df = pd.read_csv(f)
            else:
                # No headers - assign our own column names
                df = pd.read_csv(f, header=None)
                if len(df.columns) >= len(expected_cols):
                    df.columns = expected_cols[:len(df.columns)]
                elif len(df.columns) == len(expected_cols) - 1:
                    # Some files might be missing the 'ignore' column
                    df.columns = expected_cols[:len(df.columns)]
    
    # Keep needed columns
    keep_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume', 'count']
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()
    
    # Convert open_time from ms to datetime (UTC)
    # Handle both numeric timestamps and already-parsed datetime strings
    if 'open_time' in df.columns:
        sample_val = df['open_time'].iloc[0]
        if isinstance(sample_val, (int, float)) or str(sample_val).isdigit():
            # Numeric timestamp in milliseconds
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        else:
            # Already a datetime string - parse it
            df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    
    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_volume', 'taker_buy_quote_volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def read_funding_zip(path: Path) -> pd.DataFrame:
    """Read funding rate archive."""
    path = Path(path)
    
    with zipfile.ZipFile(path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV entries found inside {path}")
        
        with zf.open(csv_members[0]) as f:
            df = pd.read_csv(f)
    
    # Rename columns to standard format
    if 'calc_time' in df.columns:
        df = df.rename(columns={'calc_time': 'time'})
    elif 'fundingTime' in df.columns:
        df = df.rename(columns={'fundingTime': 'time'})
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    if 'fundingRate' in df.columns:
        df = df.rename(columns={'fundingRate': 'funding_rate'})
    
    return df


def read_aggtrades_zip(path: Path) -> pd.DataFrame:
    """
    Read aggregated trades archive.
    
    Handles both formats:
    - CSV without headers (older Binance archives)
    - CSV with headers like 'transact_time' (newer Binance archives)
    """
    path = Path(path)
    
    # Standard Binance aggTrades column names
    expected_cols = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 
                     'last_trade_id', 'time', 'is_buyer_maker']
    
    with zipfile.ZipFile(path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV entries found inside {path}")
        
        member = csv_members[0]
        
        # Peek at first line to detect header
        with zf.open(member) as f:
            first_line = f.readline().decode('utf-8').strip()
        
        # Check if first line contains text headers
        has_header = 'transact_time' in first_line.lower() or 'agg_trade_id' in first_line.lower() or not first_line.split(',')[0].isdigit()
        
        with zf.open(member) as f:
            if has_header:
                df = pd.read_csv(f)
                # Rename transact_time to time if present
                if 'transact_time' in df.columns:
                    df = df.rename(columns={'transact_time': 'time'})
            else:
                df = pd.read_csv(f, header=None)
                if len(df.columns) >= len(expected_cols):
                    df.columns = expected_cols[:len(df.columns)]
    
    # Convert time column
    if 'time' in df.columns:
        sample_val = df['time'].iloc[0]
        if isinstance(sample_val, (int, float)) or str(sample_val).isdigit():
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        else:
            df['time'] = pd.to_datetime(df['time'], utc=True)
    
    return df


def parse_klines_to_parquet(input_dir: Optional[Path] = None,
                            output_path: Optional[Path] = None,
                            pattern: str = "BTCUSDT-15m-*.zip") -> Path:
    """
    Parse all klines ZIP files into a single Parquet file.
    
    Args:
        input_dir: Directory with ZIP files
        output_path: Output Parquet path
        pattern: Glob pattern to match ZIP files
    
    Returns:
        Path to output Parquet file
    """
    base_dir = Path(__file__).parent.parent.parent
    
    if input_dir is None:
        input_dir = base_dir / "data" / "download" / "klines"
    if output_path is None:
        output_path = base_dir / "data" / "processed" / "klines_15min_all.parquet"
    
    input_files = sorted(input_dir.glob(pattern))
    
    # Also include daily files if they exist
    daily_dir = base_dir / "data" / "download" / "klines_daily"
    if daily_dir.exists():
        input_files.extend(sorted(daily_dir.glob(pattern.replace("*-*", "*"))))
    
    if not input_files:
        print(f"No kline files found matching {pattern} in {input_dir}")
        return output_path
    
    print(f"\n📊 Parsing {len(input_files)} kline files...")
    
    dfs = []
    for fp in tqdm(input_files, desc="Parsing klines"):
        try:
            df_chunk = read_klines_zip(fp)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"⚠️ Error parsing {fp}: {e}")
            continue
    
    if not dfs:
        print("No dataframes created!")
        return output_path
    
    # Combine and deduplicate
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=['open_time'])
    all_df = all_df.sort_values('open_time').reset_index(drop=True)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(output_path, index=False)
    
    print(f"✅ Saved klines to {output_path}")
    print(f"   Shape: {all_df.shape}")
    print(f"   Date range: {all_df['open_time'].min()} to {all_df['open_time'].max()}")
    
    return output_path


def parse_fundings_to_parquet(input_dir: Optional[Path] = None,
                              output_path: Optional[Path] = None) -> Path:
    """Parse all funding rate ZIP files into Parquet."""
    base_dir = Path(__file__).parent.parent.parent
    
    if input_dir is None:
        input_dir = base_dir / "data" / "download" / "fundings"
    if output_path is None:
        output_path = base_dir / "data" / "processed" / "fundings.parquet"
    
    input_files = sorted(input_dir.glob("*.zip"))
    
    if not input_files:
        print(f"No funding files found in {input_dir}")
        return output_path
    
    print(f"\n📊 Parsing {len(input_files)} funding files...")
    
    dfs = []
    for fp in tqdm(input_files, desc="Parsing fundings"):
        try:
            df_chunk = read_funding_zip(fp)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"⚠️ Error parsing {fp}: {e}")
            continue
    
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df = all_df.drop_duplicates(subset=['time'])
        all_df = all_df.sort_values('time').reset_index(drop=True)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(output_path, index=False)
        
        print(f"✅ Saved fundings to {output_path}")
    
    return output_path


def aggregate_trades_to_ohlcv(input_dir: Optional[Path] = None,
                               output_path: Optional[Path] = None,
                               timeframe: str = "15min") -> Path:
    """
    Aggregate raw trades into OHLCV with full bid/ask metrics.
    
    Creates volume breakdown from aggTrades with features (ported from old project):
    - bid_vol / ask_vol: total volume per side in period
    - max_bid_vol / max_ask_vol: largest single trade per side (whale detector)
    - avg_bid_vol / avg_ask_vol: average trade size per side
    - delta_vol: bid_vol - ask_vol (order flow imbalance)
    - cvd: cumulative volume delta (running sum)
    
    Terminology:
    - is_buyer_maker=True → seller was taker (aggressive sell) → "ask" side
    - is_buyer_maker=False → buyer was taker (aggressive buy) → "bid" side
    """
    base_dir = Path(__file__).parent.parent.parent
    
    if input_dir is None:
        input_dir = base_dir / "data" / "download" / "aggtrades"
    if output_path is None:
        output_path = base_dir / "data" / "processed" / f"aggtrades_{timeframe}_all.parquet"
    
    input_files = sorted(input_dir.glob("*.zip"))
    
    if not input_files:
        print(f"No aggTrades files found in {input_dir}")
        return output_path
    
    print(f"\n📊 Aggregating {len(input_files)} trade files to {timeframe}...")
    print(f"   Adding full bid/ask metrics (ported from old project)...")
    
    all_agg = []
    
    for fp in tqdm(input_files, desc="Processing trades"):
        try:
            df = read_aggtrades_zip(fp)
            
            # Create bid/ask columns based on is_buyer_maker
            # is_buyer_maker=True means SELLER was the maker → this is an ASK being hit
            # is_buyer_maker=False means BUYER was the maker → this is a BID being hit
            df['bid_vol'] = df['quantity'].where(df['is_buyer_maker'] == True, 0)
            df['ask_vol'] = df['quantity'].where(df['is_buyer_maker'] == False, 0)
            
            # Set time as index for resampling
            df = df.set_index('time')
            
            # Aggregate with full statistics
            agg = df.resample(timeframe).agg({
                'price': ['first', 'max', 'min', 'last'],
                'quantity': 'sum',
                'bid_vol': ['sum', 'max', 'mean'],
                'ask_vol': ['sum', 'max', 'mean']
            })
            
            # Flatten column names
            agg.columns = ['_'.join(col) for col in agg.columns.values]
            
            # Rename to final format
            agg = agg.rename(columns={
                'price_first': 'open',
                'price_max': 'high',
                'price_min': 'low',
                'price_last': 'close',
                'quantity_sum': 'total_vol',
                'bid_vol_sum': 'bid_vol',
                'bid_vol_max': 'max_bid_vol',
                'bid_vol_mean': 'avg_bid_vol',
                'ask_vol_sum': 'ask_vol',
                'ask_vol_max': 'max_ask_vol',
                'ask_vol_mean': 'avg_ask_vol'
            })
            
            # Volume delta (order flow imbalance)
            agg['delta_vol'] = agg['bid_vol'] - agg['ask_vol']
            
            agg = agg.reset_index()
            agg = agg.rename(columns={'time': 'datetime'})
            
            # Fill NaN with 0
            agg = agg.fillna(0)
            
            all_agg.append(agg)
            
        except Exception as e:
            print(f"⚠️ Error processing {fp}: {e}")
            continue
    
    if all_agg:
        final_df = pd.concat(all_agg, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['datetime'])
        final_df = final_df.sort_values('datetime').reset_index(drop=True)
        
        # Add CVD (Cumulative Volume Delta)
        final_df['cvd'] = final_df['delta_vol'].cumsum()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_path, index=False)
        
        print(f"✅ Saved aggregated trades to {output_path}")
        print(f"   Shape: {final_df.shape}")
        print(f"   Columns: {final_df.columns.tolist()}")
    
    return output_path


def create_merged_dataset(output_path: Optional[Path] = None) -> Path:
    """
    Create a single merged parquet file with ALL data sources combined.
    
    Merges:
    - klines (OHLCV)
    - funding rates
    - aggregated trades with bid/ask metrics
    
    This creates the final dataset ready for feature engineering.
    """
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / "data" / "processed"
    
    if output_path is None:
        output_path = processed_dir / "all_merged.parquet"
    
    print("\n" + "=" * 60)
    print("📦 CREATING MERGED DATASET")
    print("=" * 60)
    
    # Load klines
    klines_path = processed_dir / "klines_15min_all.parquet"
    if not klines_path.exists():
        raise FileNotFoundError(f"Klines not found: {klines_path}")
    
    print("\n1️⃣ Loading klines...")
    df = pd.read_parquet(klines_path)
    if 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'time'})
    df['time'] = pd.to_datetime(df['time'], utc=True)
    print(f"   Loaded {len(df):,} rows")
    
    # Load funding rates
    fundings_path = processed_dir / "fundings.parquet"
    if fundings_path.exists():
        print("\n2️⃣ Loading funding rates...")
        fundings = pd.read_parquet(fundings_path)
        if 'calc_time' in fundings.columns:
            fundings = fundings.rename(columns={'calc_time': 'time'})
        fundings['time'] = pd.to_datetime(fundings['time'], utc=True)
        
        # Merge and forward fill
        # NOTE: Column is usually 'last_funding_rate' in raw data
        funding_col = 'last_funding_rate' if 'last_funding_rate' in fundings.columns else 'funding_rate'
        
        if funding_col in fundings.columns:
            df = pd.merge(df, fundings[['time', funding_col]], on='time', how='left')
            df[funding_col] = df[funding_col].ffill()
            # Rename to standard 'funding_rate' if needed, or keep original
            print(f"   Merged funding rates (using {funding_col})")
        else:
            print(f"   ⚠️ Warning: No funding rate column found (expected {funding_col}). Columns: {fundings.columns.tolist()}")
    
    # Load aggregated trades
    aggtrades_path = processed_dir / "aggtrades_15min_all.parquet"
    if aggtrades_path.exists():
        print("\n3️⃣ Loading aggregated trades...")
        aggtrades = pd.read_parquet(aggtrades_path)
        if 'datetime' in aggtrades.columns:
            aggtrades = aggtrades.rename(columns={'datetime': 'time'})
        aggtrades['time'] = pd.to_datetime(aggtrades['time'], utc=True)
        
        # Select only trade-derived columns (exclude OHLC which comes from klines)
        trade_cols = ['time', 'bid_vol', 'max_bid_vol', 'avg_bid_vol',
                      'ask_vol', 'max_ask_vol', 'avg_ask_vol', 
                      'delta_vol', 'cvd', 'total_vol']
        available_cols = [c for c in trade_cols if c in aggtrades.columns]
        aggtrades = aggtrades[available_cols]
        
        # Merge
        df = pd.merge(df, aggtrades, on='time', how='left')
        print(f"   Merged trade metrics: {available_cols[1:]}")  # skip 'time'
    
    # Sort and clean
    df = df.sort_values('time').reset_index(drop=True)
    df = df.fillna(0)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved merged dataset to {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    return output_path


def parse_all():
    """Parse all downloaded data into Parquet files + create merged dataset."""
    import argparse
    parser = argparse.ArgumentParser(description="Parse downloaded data.")
    parser.add_argument("--merge-only", action="store_true", help="Skip parsing and only merge existing files")
    args = parser.parse_args()

    print("=" * 60)
    if args.merge_only:
         print("🔄 MERGING EXISTING DATASETS (Skipping parse steps)")
    else:
         print("🔄 Parsing all downloaded data")
    print("=" * 60)
    
    if not args.merge_only:
        parse_klines_to_parquet()
        parse_fundings_to_parquet()
        aggregate_trades_to_ohlcv()
    
    # Create merged dataset
    try:
        create_merged_dataset()
    except Exception as e:
        print(f"\n❌ Error creating merged dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Operation complete!")


if __name__ == "__main__":
    parse_all()
