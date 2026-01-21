"""
Technical Indicators for Bitcoin Price Prediction

Generates technical indicators using pandas-ta library.
Based on old_project_files/train.py add_*_category() functions.

All indicators are organized into categories:
- Momentum: RSI, MACD, ROC, Stochastic, etc.
- Overlap: Moving averages (EMA, SMA, VWAP, etc.)
- Trend: ADX, Aroon, MACD, etc.
- Volatility: ATR, Bollinger Bands, etc.
- Volume: OBV, MFI, CMF, etc.
- Statistics: ZScore, Entropy, Kurtosis, etc.
- Candle: Candlestick patterns
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
from typing import List, Dict, Callable, Optional

warnings.filterwarnings('ignore')


def add_momentum_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators.
    
    Indicators:
    - RSI (14): Relative Strength Index
    - ROC (12): Rate of Change
    - Stochastic (14, 3): Stochastic Oscillator
    - StochRSI (14): Stochastic RSI
    - CCI (14): Commodity Channel Index
    - Williams %R (14): Williams Percent Range
    - AO: Awesome Oscillator
    - MOM (10): Momentum
    - TSI (13, 25): True Strength Index
    - UO: Ultimate Oscillator
    """
    df = df.copy()
    
    df.ta.rsi(length=14, append=True)
    df.ta.roc(length=12, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.stochrsi(length=14, append=True)
    df.ta.cci(length=14, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.ao(append=True)
    df.ta.mom(length=10, append=True)
    df.ta.tsi(fast=13, slow=25, append=True)
    df.ta.uo(append=True)
    
    return df


def add_overlap_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overlap indicators (moving averages and bands).
    
    Indicators:
    - EMA (10, 20, 50, 100): Exponential Moving Averages
    - SMA (200): Simple Moving Average
    - HMA (9): Hull Moving Average
    - TEMA (9): Triple EMA
    - PSAR: Parabolic SAR
    - Supertrend (7, 3): Supertrend indicator
    - VWAP: Volume Weighted Average Price
    """
    df = df.copy()
    
    df.ta.ema(length=10, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.hma(length=9, append=True)
    df.ta.tema(length=9, append=True)
    df.ta.psar(append=True)
    df.ta.supertrend(length=7, multiplier=3, append=True)
    
    # VWAP needs volume
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
    Add trend indicators.
    
    Indicators:
    - MACD (12, 26): Moving Average Convergence Divergence
    - ADX (14): Average Directional Index
    - Aroon (14): Aroon Indicator
    - Vortex (14): Vortex Indicator
    - DPO (20): Detrended Price Oscillator
    - TRIX (30): Triple Exponential Average
    - CKSP: Chande Kroll Stop
    """
    df = df.copy()
    
    df.ta.macd(fast=12, slow=26, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.aroon(length=14, append=True)
    df.ta.vortex(length=14, append=True)
    df.ta.dpo(length=20, centered=False, append=True)
    df.ta.trix(length=30, append=True)
    df.ta.cksp(append=True)
    
    return df


def add_volatility_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators.
    
    Indicators:
    - ATR (14): Average True Range
    - NATR (14): Normalized ATR
    - UI (14): Ulcer Index
    - BBands (20): Bollinger Bands (BBP, BBB)
    - KC: Keltner Channel (KCP)
    - Donchian: Donchian Channel (DCP)
    """
    df = df.copy()
    
    df.ta.atr(length=14, append=True)
    df.ta.natr(length=14, append=True)
    df.ta.ui(length=14, append=True)
    
    # Bollinger Bands - extract key metrics
    bbands = df.ta.bbands(length=20, append=False)
    if bbands is not None and not bbands.empty:
        bbp_cols = [c for c in bbands.columns if c.startswith('BBP')]
        bbb_cols = [c for c in bbands.columns if c.startswith('BBB')]
        if bbp_cols:
            df[bbp_cols[0]] = bbands[bbp_cols[0]]
        if bbb_cols:
            df[bbb_cols[0]] = bbands[bbb_cols[0]]
    
    # Keltner Channel
    kc = df.ta.kc(append=False)
    if kc is not None and not kc.empty:
        kcp_cols = [c for c in kc.columns if c.startswith('KCP')]
        if kcp_cols:
            df[kcp_cols[0]] = kc[kcp_cols[0]]
    
    # Donchian Channel
    dc = df.ta.donchian(append=False)
    if dc is not None and not dc.empty:
        dcp_cols = [c for c in dc.columns if c.startswith('DCP')]
        if dcp_cols:
            df[dcp_cols[0]] = dc[dcp_cols[0]]
    
    return df


def add_volume_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume indicators.
    
    Indicators:
    - OBV: On-Balance Volume
    - MFI (14): Money Flow Index
    - AD: Accumulation/Distribution
    - CMF: Chaikin Money Flow
    - EOM: Ease of Movement
    - NVI: Negative Volume Index
    - PVI: Positive Volume Index
    """
    df = df.copy()
    
    vol_col = 'volume' if 'volume' in df.columns else 'total_vol'
    
    if vol_col in df.columns:
        df.ta.obv(close=df['close'], volume=df[vol_col], append=True)
        df.ta.mfi(high=df['high'], low=df['low'], close=df['close'], 
                 volume=df[vol_col], length=14, append=True)
        df.ta.ad(high=df['high'], low=df['low'], close=df['close'], 
                volume=df[vol_col], append=True)
        df.ta.cmf(high=df['high'], low=df['low'], close=df['close'], 
                 volume=df[vol_col], append=True)
        df.ta.eom(high=df['high'], low=df['low'], close=df['close'], 
                 volume=df[vol_col], append=True)
        df.ta.nvi(close=df['close'], volume=df[vol_col], append=True)
        df.ta.pvi(close=df['close'], volume=df[vol_col], append=True)
    
    return df


def add_statistics_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add statistical indicators.
    
    Indicators:
    - ZScore (30): Z-Score
    - Entropy (30): Entropy
    - Kurtosis (30): Kurtosis
    - Skew (30): Skewness
    - Variance (30): Variance
    - MAD (30): Mean Absolute Deviation
    """
    df = df.copy()
    
    df.ta.zscore(length=30, append=True)
    df.ta.entropy(length=30, append=True)
    df.ta.kurtosis(length=30, append=True)
    df.ta.skew(length=30, append=True)
    df.ta.variance(length=30, append=True)
    df.ta.mad(length=30, append=True)
    
    return df


def add_candle_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick pattern indicators.
    
    Patterns:
    - Doji, Hammer, Engulfing, Morning/Evening Star
    - Shooting Star, Hanging Man, Marubozu
    - Three White Soldiers, Three Black Crows
    - Inside Bar, Spinning Top
    """
    df = df.copy()
    
    patterns_to_add = [
        'cdl_doji', 'cdl_hammer', 'cdl_engulfing', 
        'cdl_morningstar', 'cdl_eveningstar',
        'cdl_shootingstar', 'cdl_hangingman', 'cdl_marubozu',
        'cdl_3whitesoldiers', 'cdl_3blackcrows',
        'cdl_inside', 'cdl_spinningtop'
    ]
    
    for pattern_name in patterns_to_add:
        if hasattr(df.ta, pattern_name):
            try:
                getattr(df.ta, pattern_name)(append=True)
            except Exception:
                pass  # Some patterns may fail
    
    return df


# Dictionary mapping group names to functions
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
    Add indicators from specified groups.
    
    Args:
        df: DataFrame with OHLCV data
        groups: List of group names to add. If None, adds all groups.
    
    Returns:
        DataFrame with added indicator columns
    """
    if groups is None:
        groups = list(GROUP_FUNCS.keys())
    
    df_result = df.copy()
    
    for group_name in groups:
        if group_name in GROUP_FUNCS:
            print(f"  Adding {group_name} indicators...")
            df_result = GROUP_FUNCS[group_name](df_result)
    
    return df_result


def get_indicator_columns(df: pd.DataFrame,
                          exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get list of indicator columns (excluding base OHLCV and target).
    
    Args:
        df: DataFrame with indicators
        exclude_cols: Additional columns to exclude
    
    Returns:
        List of indicator column names
    """
    base_cols = {'time', 'open', 'high', 'low', 'close', 'volume',
                 'total_vol', 'buy_vol', 'sell_vol', 'quote_volume',
                 'taker_buy_volume', 'taker_buy_quote_volume', 'count',
                 'funding_rate', 'log_return',
                 'target', 'smoothed_close', 'smooth_slope'}
    
    if exclude_cols:
        base_cols.update(exclude_cols)
    
    indicator_cols = [c for c in df.columns if c not in base_cols]
    
    return indicator_cols


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='15min'),
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 101,
        'low': np.random.randn(n).cumsum() + 99,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.rand(n) * 1000
    })
    
    # Fix OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    print("Adding all indicators...")
    df_indicators = add_all_indicators(df)
    
    indicator_cols = get_indicator_columns(df_indicators)
    
    print(f"\nTotal columns: {len(df_indicators.columns)}")
    print(f"Indicator columns: {len(indicator_cols)}")
    print(f"\nIndicator names:\n{indicator_cols[:20]}...")
