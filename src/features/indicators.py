"""
Technical Indicators Module for Bitcoin Price Prediction.

This module generates 60+ technical indicators using the pandas-ta library.
All indicators are organized into categories for easy selection and analysis.

Indicator Categories:
--------------------
1. Momentum   - RSI, MACD, Stochastic, etc. (measure speed of price changes)
2. Overlap    - Moving averages (EMA, SMA, VWAP) that overlay on price
3. Trend      - ADX, Aroon, MACD (identify trend direction/strength)
4. Volatility - ATR, Bollinger Bands (measure price variability)
5. Volume     - OBV, MFI, CMF (analyze trading volume patterns)
6. Statistics - ZScore, Entropy, Kurtosis (statistical measures)
7. Candle     - Candlestick patterns (doji, hammer, engulfing, etc.)

Usage Example:
-------------
    from src.features.indicators import add_all_indicators, GROUP_FUNCS
    
    # Add all indicators
    df_with_indicators = add_all_indicators(df)
    
    # Add only specific groups
    df = add_momentum_category(df)
    df = add_volatility_category(df)

Notes:
------
- All indicators use pandas-ta library (130+ built-in indicators)
- Indicators are calculated on OHLCV data
- NaN values at the beginning are normal (warmup period)
- Always shift indicators by prediction horizon before training!

Author: Capstone Project
Date: 2026
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
from typing import List, Dict, Callable, Optional

# Suppress pandas-ta warnings about missing columns
warnings.filterwarnings('ignore')


def add_momentum_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators to DataFrame.
    
    Momentum indicators measure the speed/velocity of price changes.
    They help identify overbought/oversold conditions and potential reversals.
    
    Indicators Added:
    -----------------
    - RSI (14): Relative Strength Index - values 0-100, >70 overbought, <30 oversold
    - ROC (12): Rate of Change - percentage change over N periods
    - Stochastic (14,3): Compares close to high-low range
    - StochRSI (14): RSI applied to RSI values
    - CCI (14): Commodity Channel Index - deviation from mean
    - Williams %R (14): Similar to Stochastic, inverted scale
    - AO: Awesome Oscillator - 5-period vs 34-period SMA difference
    - MOM (10): Simple momentum (current price - price N bars ago)
    - TSI (13,25): True Strength Index - double-smoothed momentum
    - UO: Ultimate Oscillator - weighted average of 3 timeframes
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns (open, high, low, close, volume).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with momentum indicator columns added.
    """
    df = df.copy()
    
    # RSI - most popular momentum indicator
    # Values range 0-100, typically: >70 = overbought, <30 = oversold
    df.ta.rsi(length=14, append=True)
    
    # Rate of Change - percentage change over 12 periods
    df.ta.roc(length=12, append=True)
    
    # Stochastic Oscillator - where close is relative to high-low range
    # Returns STOCHk (fast) and STOCHd (slow/signal)
    df.ta.stoch(k=14, d=3, append=True)
    
    # Stochastic RSI - applies stochastic formula to RSI values
    df.ta.stochrsi(length=14, append=True)
    
    # Commodity Channel Index - deviation from statistical mean
    df.ta.cci(length=14, append=True)
    
    # Williams %R - similar to stochastic but inverted (-100 to 0)
    df.ta.willr(length=14, append=True)
    
    # Awesome Oscillator - 5-period SMA minus 34-period SMA of midpoint
    df.ta.ao(append=True)
    
    # Momentum - simply close minus close N periods ago
    df.ta.mom(length=10, append=True)
    
    # True Strength Index - double-smoothed price momentum
    df.ta.tsi(fast=13, slow=25, append=True)
    
    # Ultimate Oscillator - weighted sum of 3 different periods
    df.ta.uo(append=True)
    
    return df


def add_overlap_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overlap indicators (moving averages) to DataFrame.
    
    Overlap indicators are plotted on the same scale as price.
    They help identify trends and support/resistance levels.
    
    Indicators Added:
    -----------------
    - EMA (10,20,50,100): Exponential Moving Averages at various lengths
    - SMA (200): Simple Moving Average - classic long-term trend indicator
    - HMA (9): Hull Moving Average - faster response, less lag
    - TEMA (9): Triple EMA - even less lag than EMA
    - PSAR: Parabolic SAR - trailing stop/trend indicator
    - Supertrend (7,3): Popular trend-following indicator
    - VWAP: Volume Weighted Average Price (if volume available)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with overlap indicator columns added.
    """
    df = df.copy()
    
    # Exponential Moving Averages at multiple timeframes
    # Shorter EMAs react faster, longer EMAs show bigger trends
    df.ta.ema(length=10, append=True)   # Fast EMA
    df.ta.ema(length=20, append=True)   # Medium EMA
    df.ta.ema(length=50, append=True)   # Slow EMA
    df.ta.ema(length=100, append=True)  # Very slow EMA
    
    # Simple Moving Average - equal weight to all periods
    df.ta.sma(length=200, append=True)  # Classic long-term trend
    
    # Hull Moving Average - responsive with reduced lag
    df.ta.hma(length=9, append=True)
    
    # Triple Exponential Moving Average - even less lag
    df.ta.tema(length=9, append=True)
    
    # NOTE: Parabolic SAR and Supertrend removed - they produce 30000+ NaN
    # because they only output values during trend changes, not every bar
    # This corrupts the training data when filled with zeros
    
    # VWAP - Volume Weighted Average Price
    # Only calculated if volume is available
    vol_col = 'volume' if 'volume' in df.columns else 'total_vol'
    if vol_col in df.columns:
        try:
            df.ta.vwap(high=df['high'], low=df['low'], close=df['close'], 
                      volume=df[vol_col], append=True)
        except Exception:
            pass  # VWAP may fail if index is not DatetimeIndex
    
    return df


def add_trend_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators to DataFrame.
    
    Trend indicators help identify the direction and strength of trends.
    
    Indicators Added:
    -----------------
    - MACD (12,26): Moving Average Convergence Divergence - trend momentum
    - ADX (14): Average Directional Index - trend strength (0-100)
    - Aroon (14): Identifies trend changes and strength
    - Vortex (14): Positive and negative trend movement
    - DPO (20): Detrended Price Oscillator - removes trend to show cycles
    - TRIX (30): Triple exponential smoothed rate of change
    - CKSP: Chande Kroll Stop - adaptive stop levels
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with trend indicator columns added.
    """
    df = df.copy()
    
    # MACD - the most popular trend/momentum indicator
    # Returns MACD line, Signal line, and Histogram
    df.ta.macd(fast=12, slow=26, append=True)
    
    # ADX - measures trend STRENGTH (not direction)
    # ADX > 25 = strong trend, < 20 = weak/no trend
    df.ta.adx(length=14, append=True)
    
    # Aroon - identifies when trends are likely to change
    df.ta.aroon(length=14, append=True)
    
    # Vortex Indicator - positive and negative trend movement
    df.ta.vortex(length=14, append=True)
    
    # Detrended Price Oscillator - removes trend, shows cycles
    df.ta.dpo(length=20, centered=False, append=True)
    
    # TRIX - triple smoothed rate of change, filters noise
    df.ta.trix(length=30, append=True)
    
    # Chande Kroll Stop - adaptive stop loss levels
    df.ta.cksp(append=True)
    
    return df


def add_volatility_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators to DataFrame.
    
    Volatility indicators measure the degree of price variation.
    High volatility = large price swings, low volatility = stable prices.
    
    Indicators Added:
    -----------------
    - ATR (14): Average True Range - absolute volatility measure
    - NATR (14): Normalized ATR - ATR as percentage of price
    - UI (14): Ulcer Index - downside volatility/risk
    - BBands (20): Bollinger Bands - standard deviation bands
    - KC: Keltner Channel - ATR-based bands
    - Donchian: Donchian Channel - high/low range
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with volatility indicator columns added.
    """
    df = df.copy()
    
    # Average True Range - the standard volatility measure
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    df.ta.atr(length=14, append=True)
    
    # Normalized ATR - ATR as percentage of close price
    # Useful for comparing volatility across different price levels
    df.ta.natr(length=14, append=True)
    
    # Ulcer Index - measures downside risk/volatility
    df.ta.ui(length=14, append=True)
    
    # Bollinger Bands - we extract key metrics
    # BBP = Bollinger Band Percentage (where price is within bands)
    # BBB = Bollinger Bandwidth (distance between bands)
    bbands = df.ta.bbands(length=20, append=False)
    if bbands is not None and not bbands.empty:
        # Find the BBP (percent) and BBB (bandwidth) columns
        bbp_cols = [c for c in bbands.columns if c.startswith('BBP')]
        bbb_cols = [c for c in bbands.columns if c.startswith('BBB')]
        if bbp_cols:
            df[bbp_cols[0]] = bbands[bbp_cols[0]]
        if bbb_cols:
            df[bbb_cols[0]] = bbands[bbb_cols[0]]
    
    # Keltner Channel - ATR-based bands (similar to Bollinger)
    kc = df.ta.kc(append=False)
    if kc is not None and not kc.empty:
        kcp_cols = [c for c in kc.columns if c.startswith('KCP')]
        if kcp_cols:
            df[kcp_cols[0]] = kc[kcp_cols[0]]
    
    # Donchian Channel - simple high/low range bands
    dc = df.ta.donchian(append=False)
    if dc is not None and not dc.empty:
        dcp_cols = [c for c in dc.columns if c.startswith('DCP')]
        if dcp_cols:
            df[dcp_cols[0]] = dc[dcp_cols[0]]
    
    return df


def add_volume_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based indicators to DataFrame.
    
    Volume indicators analyze trading volume to confirm trends
    and identify potential reversals.
    
    Indicators Added:
    -----------------
    - OBV: On-Balance Volume - cumulative volume based on direction
    - MFI (14): Money Flow Index - volume-weighted RSI
    - AD: Accumulation/Distribution - money flow into/out of asset
    - CMF: Chaikin Money Flow - buying vs selling pressure
    - EOM: Ease of Movement - price change vs volume
    - NVI: Negative Volume Index - tracks low-volume days
    - PVI: Positive Volume Index - tracks high-volume days
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns. Must have volume column.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with volume indicator columns added.
    """
    df = df.copy()
    
    # Determine which volume column exists
    vol_col = 'volume' if 'volume' in df.columns else 'total_vol'
    
    if vol_col not in df.columns:
        # No volume data available, skip volume indicators
        return df
    
    # On-Balance Volume - adds volume on up days, subtracts on down days
    df.ta.obv(close=df['close'], volume=df[vol_col], append=True)
    
    # Money Flow Index - essentially RSI but volume-weighted
    df.ta.mfi(high=df['high'], low=df['low'], close=df['close'], 
             volume=df[vol_col], length=14, append=True)
    
    # Accumulation/Distribution - measures money flow
    df.ta.ad(high=df['high'], low=df['low'], close=df['close'], 
            volume=df[vol_col], append=True)
    
    # Chaikin Money Flow - measures buying vs selling pressure
    df.ta.cmf(high=df['high'], low=df['low'], close=df['close'], 
             volume=df[vol_col], append=True)
    
    # Ease of Movement - how easily price moves based on volume
    df.ta.eom(high=df['high'], low=df['low'], close=df['close'], 
             volume=df[vol_col], append=True)
    
    # Negative Volume Index - price changes on low-volume days
    df.ta.nvi(close=df['close'], volume=df[vol_col], append=True)
    
    # Positive Volume Index - price changes on high-volume days
    df.ta.pvi(close=df['close'], volume=df[vol_col], append=True)
    
    return df


def add_statistics_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add statistical indicators to DataFrame.
    
    These indicators apply statistical measures to price data
    to identify unusual behavior or patterns.
    
    Indicators Added:
    -----------------
    - ZScore (30): Standard deviations from mean - identify extremes
    - Entropy (30): Information entropy - measure of randomness
    - Kurtosis (30): "Fat tails" - likelihood of extreme moves
    - Skew (30): Asymmetry of distribution
    - Variance (30): Squared deviation from mean
    - MAD (30): Mean Absolute Deviation - robust volatility
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with statistical indicator columns added.
    """
    df = df.copy()
    
    # Z-Score - how many standard deviations from mean
    # Useful for identifying overbought/oversold conditions
    df.ta.zscore(length=30, append=True)
    
    # Entropy - measure of unpredictability/randomness
    df.ta.entropy(length=30, append=True)
    
    # Kurtosis - measures "fat tails" in distribution
    # High kurtosis = more extreme moves likely
    df.ta.kurtosis(length=30, append=True)
    
    # Skewness - asymmetry of the distribution
    df.ta.skew(length=30, append=True)
    
    # Variance - squared deviation from mean
    df.ta.variance(length=30, append=True)
    
    # Mean Absolute Deviation - robust alternative to std dev
    df.ta.mad(length=30, append=True)
    
    return df


def add_candle_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick pattern indicators to DataFrame.
    
    Candlestick patterns are visual patterns in OHLC data that
    may signal reversals or continuations.
    
    Patterns Added:
    ---------------
    - Doji: Open and close are equal - indecision
    - Hammer: Small body at top, long lower wick - bullish reversal
    - Engulfing: Current candle engulfs previous - reversal
    - Morning/Evening Star: 3-candle reversal patterns
    - Shooting Star: Small body at bottom, long upper wick - bearish
    - Hanging Man: Like hammer but in uptrend - bearish
    - Marubozu: Full body, no wicks - strong move
    - Three White Soldiers/Black Crows: 3 consecutive strong candles
    - Inside Bar: Current bar within previous bar's range
    - Spinning Top: Small body, equal wicks - indecision
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with candlestick pattern columns added.
        Pattern values: 100 = bullish, -100 = bearish, 0 = no pattern
    """
    df = df.copy()
    
    # List of candlestick patterns to detect
    patterns_to_add = [
        'cdl_doji',           # Indecision pattern
        'cdl_hammer',         # Bullish reversal
        'cdl_engulfing',      # Engulfing pattern
        'cdl_morningstar',    # Bullish 3-candle reversal
        'cdl_eveningstar',    # Bearish 3-candle reversal
        'cdl_shootingstar',   # Bearish reversal
        'cdl_hangingman',     # Bearish reversal
        'cdl_marubozu',       # Strong directional move
        'cdl_3whitesoldiers', # Strong bullish continuation
        'cdl_3blackcrows',    # Strong bearish continuation
        'cdl_inside',         # Inside bar (consolidation)
        'cdl_spinningtop'     # Indecision
    ]
    
    # Add each pattern if available in pandas-ta
    for pattern_name in patterns_to_add:
        if hasattr(df.ta, pattern_name):
            try:
                getattr(df.ta, pattern_name)(append=True)
            except Exception:
                pass  # Some patterns may fail on edge cases
    
    return df


# =============================================================================
# Dictionary mapping group names to their functions
# =============================================================================
GROUP_FUNCS: Dict[str, Callable] = {
    "momentum": add_momentum_category,
    "overlap": add_overlap_category,
    "trend": add_trend_category,
    "volatility": add_volatility_category,
    "volume": add_volume_category,
    "statistics": add_statistics_category,
    "candle": add_candle_category,
}


def add_all_indicators(df: pd.DataFrame,
                       groups: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add indicators from specified groups to DataFrame.
    
    This is the main function to add technical indicators.
    By default, it adds ALL available indicator groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns.
    groups : List[str], optional
        List of group names to add. If None, adds all groups.
        Valid groups: 'momentum', 'overlap', 'trend', 'volatility',
                      'volume', 'statistics', 'candle'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all requested indicator columns added.
    
    Examples
    --------
    >>> # Add all indicators
    >>> df = add_all_indicators(df)
    
    >>> # Add only momentum and trend
    >>> df = add_all_indicators(df, groups=['momentum', 'trend'])
    """
    if groups is None:
        groups = list(GROUP_FUNCS.keys())
    
    df_result = df.copy()
    
    for group_name in groups:
        if group_name in GROUP_FUNCS:
            print(f"  Adding {group_name} indicators...")
            df_result = GROUP_FUNCS[group_name](df_result)
        else:
            print(f"  Warning: Unknown group '{group_name}'")
    
    return df_result


def get_indicator_columns(df: pd.DataFrame,
                          exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get list of indicator columns (excluding base OHLCV data).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with indicators added.
    exclude_cols : List[str], optional
        Additional columns to exclude from the result.
    
    Returns
    -------
    List[str]
        List of indicator column names.
    """
    # Base columns that are NOT indicators (raw data columns)
    # NOTE: buy_vol, sell_vol, total_vol are included as features (not excluded)
    # because they provide important volume breakdown information
    base_cols = {
        'time', 'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume', 'count',
        'funding_rate', 'log_return',
        'target', 'smoothed_close', 'smooth_slope'
    }
    
    if exclude_cols:
        base_cols.update(exclude_cols)
    
    # Everything else is an indicator
    indicator_cols = [c for c in df.columns if c not in base_cols]
    
    return indicator_cols


# =============================================================================
# Module Test
# =============================================================================
if __name__ == "__main__":
    """Test indicator generation with dummy data."""
    
    print("=" * 60)
    print("Testing Technical Indicators Module")
    print("=" * 60)
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 500
    
    # Generate realistic-ish OHLC data
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='15min'),
        'open': close_prices + np.random.randn(n) * 0.3,
        'high': close_prices + np.abs(np.random.randn(n) * 0.5),
        'low': close_prices - np.abs(np.random.randn(n) * 0.5),
        'close': close_prices,
        'volume': np.random.rand(n) * 1000 + 100
    })
    
    # Ensure OHLC consistency (high >= all, low <= all)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    print(f"\nInput DataFrame shape: {df.shape}")
    
    # Add all indicators
    print("\nAdding indicators...")
    df_indicators = add_all_indicators(df)
    
    # Get indicator columns
    indicator_cols = get_indicator_columns(df_indicators)
    
    print(f"\nOutput DataFrame shape: {df_indicators.shape}")
    print(f"Number of indicators added: {len(indicator_cols)}")
    print(f"\nFirst 20 indicator names:")
    for i, col in enumerate(indicator_cols[:20]):
        print(f"  {i+1}. {col}")
    
    print("\n✅ Indicators module test complete!")
