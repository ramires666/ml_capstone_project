"""
Data Downloader for Binance Futures BTCUSDT

Downloads historical klines from data.binance.vision archives (monthly + daily).
Based on old_project_files/data/downloader.py with improvements.
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import zipfile
import io


# --- Configuration ---
BASE_URL_MONTHLY = "https://data.binance.vision/data/futures/um/monthly"
BASE_URL_DAILY = "https://data.binance.vision/data/futures/um/daily"


def generate_monthly_dates(start_year: int, start_month: int, 
                           end_year: int, end_month: int) -> List[Tuple[int, str]]:
    """Generate year and month pairs from start to end date."""
    dates = []
    current_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)
    
    while current_date <= end_date:
        dates.append((current_date.year, f"{current_date.month:02d}"))
        # Move to next month
        current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
    
    return dates


def generate_daily_dates(start_date: datetime, end_date: datetime) -> List[str]:
    """Generate daily date strings from start to end."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def download_file(url: str, destination_path: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded: {destination_path.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        if "404" in str(e):
            print(f"✗ Not found: {destination_path.name}")
        else:
            print(f"✗ Error downloading {url}: {e}")
        return False


def download_monthly_klines(symbol: str = "BTCUSDT", 
                            timeframe: str = "15m",
                            start_year: int = 2024, 
                            start_month: int = 1,
                            end_year: int = 2025, 
                            end_month: int = 12,
                            dest_folder: Optional[Path] = None) -> List[Path]:
    """
    Download monthly klines archives from Binance Vision.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        timeframe: Kline interval (e.g., 15m, 1h)
        start_year, start_month: Start date
        end_year, end_month: End date
        dest_folder: Destination folder (default: data/download/klines)
    
    Returns:
        List of downloaded file paths
    """
    if dest_folder is None:
        dest_folder = Path(__file__).parent.parent.parent / "data" / "download" / "klines"
    
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    dates = generate_monthly_dates(start_year, start_month, end_year, end_month)
    
    print(f"\n📥 Downloading monthly klines for {symbol} ({timeframe})")
    print(f"   Period: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    
    for year, month in dates:
        filename = f"{symbol}-{timeframe}-{year}-{month}.zip"
        url = f"{BASE_URL_MONTHLY}/klines/{symbol}/{timeframe}/{filename}"
        dest_path = dest_folder / filename
        
        if dest_path.exists():
            print(f"⏭ Skipping (exists): {filename}")
            downloaded.append(dest_path)
            continue
        
        if download_file(url, dest_path):
            downloaded.append(dest_path)
    
    print(f"\n✅ Downloaded {len(downloaded)} monthly files")
    return downloaded


def download_daily_klines(symbol: str = "BTCUSDT",
                          timeframe: str = "15m", 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          dest_folder: Optional[Path] = None) -> List[Path]:
    """
    Download daily klines archives for recent data.
    
    Useful for getting data from last few days that may not be in monthly archives.
    """
    if dest_folder is None:
        dest_folder = Path(__file__).parent.parent.parent / "data" / "download" / "klines_daily"
    
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
    
    downloaded = []
    dates = generate_daily_dates(start_date, end_date)
    
    print(f"\n📥 Downloading daily klines for {symbol} ({timeframe})")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    
    for date_str in dates:
        filename = f"{symbol}-{timeframe}-{date_str}.zip"
        url = f"{BASE_URL_DAILY}/klines/{symbol}/{timeframe}/{filename}"
        dest_path = dest_folder / filename
        
        if dest_path.exists():
            print(f"⏭ Skipping (exists): {filename}")
            downloaded.append(dest_path)
            continue
        
        if download_file(url, dest_path):
            downloaded.append(dest_path)
    
    print(f"\n✅ Downloaded {len(downloaded)} daily files")
    return downloaded


def download_funding_rates(symbol: str = "BTCUSDT",
                           start_year: int = 2024,
                           start_month: int = 1,
                           end_year: int = 2025,
                           end_month: int = 12,
                           dest_folder: Optional[Path] = None) -> List[Path]:
    """Download monthly funding rate archives."""
    if dest_folder is None:
        dest_folder = Path(__file__).parent.parent.parent / "data" / "download" / "fundings"
    
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    dates = generate_monthly_dates(start_year, start_month, end_year, end_month)
    
    print(f"\n📥 Downloading funding rates for {symbol}")
    
    for year, month in dates:
        filename = f"{symbol}-fundingRate-{year}-{month}.zip"
        url = f"{BASE_URL_MONTHLY}/fundingRate/{symbol}/{filename}"
        dest_path = dest_folder / filename
        
        if dest_path.exists():
            print(f"⏭ Skipping (exists): {filename}")
            downloaded.append(dest_path)
            continue
        
        if download_file(url, dest_path):
            downloaded.append(dest_path)
    
    return downloaded


def download_agg_trades(symbol: str = "BTCUSDT",
                        start_year: int = 2024,
                        start_month: int = 1,
                        end_year: int = 2025,
                        end_month: int = 12,
                        dest_folder: Optional[Path] = None) -> List[Path]:
    """Download monthly aggregated trades archives."""
    if dest_folder is None:
        dest_folder = Path(__file__).parent.parent.parent / "data" / "download" / "aggtrades"
    
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    dates = generate_monthly_dates(start_year, start_month, end_year, end_month)
    
    print(f"\n📥 Downloading aggregated trades for {symbol}")
    
    for year, month in dates:
        filename = f"{symbol}-aggTrades-{year}-{month}.zip"
        url = f"{BASE_URL_MONTHLY}/aggTrades/{symbol}/{filename}"
        dest_path = dest_folder / filename
        
        if dest_path.exists():
            print(f"⏭ Skipping (exists): {filename}")
            downloaded.append(dest_path)
            continue
        
        if download_file(url, dest_path):
            downloaded.append(dest_path)
    
    return downloaded


def download_all(symbol: str = "BTCUSDT",
                 timeframe: str = "15m",
                 start_year: int = 2024,
                 start_month: int = 1,
                 end_year: int = 2025,
                 end_month: int = 12) -> dict:
    """
    Download all data types: klines, funding rates, and aggregated trades.
    
    Returns:
        Dict with lists of downloaded file paths for each data type.
    """
    print("=" * 60)
    print(f"🚀 Starting full data download for {symbol}")
    print("=" * 60)
    
    results = {
        "klines": download_monthly_klines(symbol, timeframe, start_year, start_month, end_year, end_month),
        "fundings": download_funding_rates(symbol, start_year, start_month, end_year, end_month),
        "aggtrades": download_agg_trades(symbol, start_year, start_month, end_year, end_month),
    }
    
    # Also get recent daily data
    results["klines_daily"] = download_daily_klines(symbol, timeframe)
    
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    for dtype, files in results.items():
        print(f"   {dtype}: {len(files)} files")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Download data for the project
    download_all(
        symbol="BTCUSDT",
        timeframe="15m",
        start_year=2024,
        start_month=1,
        end_year=2025,
        end_month=12
    )
