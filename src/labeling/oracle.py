"""
Oracle Labels Generator for Bitcoin Price Direction Prediction.

This module creates target labels using Gaussian smoothing (Oracle approach).
The "oracle" approach uses centered Gaussian smoothing to identify the true
underlying trend, then classifies each point as UP/DOWN/SIDEWAYS based on slope.

Key Concepts:
-------------
1. Apply Gaussian filter to log(close) prices to get smoothed "true" trend
2. Calculate the slope (log-difference) of the smoothed line
3. Classify based on threshold:
   - slope > threshold  -> UP (2)
   - slope < -threshold -> DOWN (0)
   - otherwise          -> SIDEWAYS (1)

CRITICAL WARNING:
-----------------
The oracle smoothed line is used ONLY for generating labels, NOT as a feature!
Using it as a feature would cause look-ahead bias (data leakage) because the
Gaussian filter uses future data points in its calculation.

Example Usage:
--------------
    from src.labeling.oracle import create_oracle_labels
    
    df_labeled = create_oracle_labels(df, sigma=4, threshold=0.0004)
    # Now df_labeled has 'target' column with values 0, 1, or 2

Author: Capstone Project
Date: 2026
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional
from numba import jit


@jit(nopython=True, cache=True)
def classify_slope_numba(slope: np.ndarray, threshold: float) -> np.ndarray:
    """
    Numba-accelerated classification of slope values into discrete labels.
    
    This function is JIT-compiled by Numba for ~10-100x speedup over pure Python.
    It classifies each slope value into one of three categories based on threshold.
    
    Parameters
    ----------
    slope : np.ndarray
        Array of slope values (log-differences of smoothed prices).
        Positive values indicate upward trend, negative indicate downward.
    threshold : float
        The minimum absolute slope value to classify as UP or DOWN.
        Typical values: 0.0003 to 0.0006 for 15-minute BTC data.
    
    Returns
    -------
    np.ndarray
        Array of integer labels:
        - 0 = DOWN  (slope < -threshold)
        - 1 = SIDEWAYS (|slope| <= threshold)
        - 2 = UP (slope > threshold)
    
    Notes
    -----
    Using Numba's nopython mode ensures maximum performance by avoiding
    Python interpreter overhead. The cache=True flag stores compiled
    code for faster subsequent runs.
    """
    # Get the number of samples to process
    n = len(slope)
    
    # Initialize all labels as SIDEWAYS (1) - the default/neutral class
    labels = np.ones(n, dtype=np.int32)
    
    # Iterate through each slope value and classify
    for i in range(n):
        if slope[i] > threshold:
            # Positive slope above threshold = UP trend
            labels[i] = 2
        elif slope[i] < -threshold:
            # Negative slope below threshold = DOWN trend
            labels[i] = 0
        # else: remains SIDEWAYS (1)
    
    return labels


def create_oracle_labels(df: pd.DataFrame,
                         sigma: int = 4,
                         threshold: float = 0.0004,
                         price_col: str = 'close') -> pd.DataFrame:
    """
    Create target labels using Gaussian smoothing (Oracle approach).
    
    This is the core labeling function that generates classification targets
    for the price direction prediction models. It uses "oracle" knowledge
    (centered smoothing) to identify the true trend at each point.
    
    Algorithm Steps:
    ----------------
    1. Take log of close prices (log-returns are more stationary)
    2. Apply centered Gaussian smoothing to get "true" trend line
    3. Calculate slope as difference of consecutive smoothed values
    4. Classify slope into UP/DOWN/SIDEWAYS based on threshold
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OHLCV data. Must have the price column specified.
    sigma : int, default=4
        Standard deviation for Gaussian kernel. Controls smoothing intensity.
        - Higher sigma = more smoothing = fewer direction changes
        - Lower sigma = less smoothing = more frequent direction changes
        - Typical range: 2-10 for 15-minute candles
    threshold : float, default=0.0004
        Slope threshold for UP/DOWN classification.
        - Higher threshold = more SIDEWAYS labels
        - Lower threshold = more UP/DOWN labels
        - Typical range: 0.0002-0.0008 for 15-minute BTC data
    price_col : str, default='close'
        Name of the column containing price data.
    
    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with additional columns:
        - 'target': int (0=DOWN, 1=SIDEWAYS, 2=UP) - USE THIS FOR TRAINING
        - 'smoothed_close': Smoothed price (FOR VISUALIZATION ONLY!)
        - 'smooth_slope': Slope value (FOR VISUALIZATION ONLY!)
    
    Warnings
    --------
    DO NOT use 'smoothed_close' or 'smooth_slope' as features!
    They contain future information and will cause data leakage.
    
    Examples
    --------
    >>> df = load_ohlcv_data()
    >>> df_labeled = create_oracle_labels(df, sigma=4, threshold=0.0004)
    >>> print(df_labeled['target'].value_counts(normalize=True))
    1    0.35  # SIDEWAYS
    2    0.33  # UP
    0    0.32  # DOWN
    """
    # Create a copy to avoid modifying the original DataFrame
    df_result = df.copy()
    
    # Extract close prices as numpy array for efficient computation
    close = df_result[price_col].values.astype(np.float64)
    
    # Step 1: Convert to log space
    # Log prices have better statistical properties (more stationary)
    log_close = np.log(close)
    
    # Step 2: Apply Gaussian smoothing to log prices
    # mode='nearest' extends edge values to avoid edge effects
    # This is a CENTERED filter, meaning it uses both past and future values!
    smoothed_log = gaussian_filter1d(log_close, sigma=sigma, mode='nearest')
    
    # Convert back to price space for visualization (not used in training!)
    df_result['smoothed_close'] = np.exp(smoothed_log)
    
    # Step 3: Calculate slope (log-difference of smoothed line)
    # prepend first value to maintain array length
    smooth_slope = np.diff(smoothed_log, prepend=smoothed_log[0])
    df_result['smooth_slope'] = smooth_slope
    
    # Step 4: Classify using Numba-accelerated function
    labels = classify_slope_numba(smooth_slope, threshold)
    df_result['target'] = labels
    
    return df_result


def analyze_label_distribution(df: pd.DataFrame,
                               sigma_range: Tuple[int, int] = (1, 10),
                               threshold_range: Tuple[float, float] = (0.0001, 0.001),
                               n_steps: int = 5) -> pd.DataFrame:
    """
    Analyze target distribution for different sigma and threshold values.
    
    This is a hyperparameter tuning helper that shows how different
    combinations of sigma and threshold affect class balance. Use this
    to find parameters that give relatively balanced class distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data.
    sigma_range : Tuple[int, int], default=(1, 10)
        Range of sigma values to test (min, max).
    threshold_range : Tuple[float, float], default=(0.0001, 0.001)
        Range of threshold values to test (min, max).
    n_steps : int, default=5
        Number of steps for each parameter in the grid search.
    
    Returns
    -------
    pd.DataFrame
        Results sorted by imbalance (lowest first), with columns:
        - sigma: Gaussian sigma value
        - threshold: Classification threshold
        - down_pct: Percentage of DOWN labels
        - sideways_pct: Percentage of SIDEWAYS labels
        - up_pct: Percentage of UP labels
        - imbalance: Absolute difference between UP and DOWN percentages
    
    Examples
    --------
    >>> results = analyze_label_distribution(df)
    >>> print(results.head())  # Show most balanced configurations
    """
    results = []
    
    # Create parameter grid
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_steps, dtype=int)
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    
    # Test each combination
    for sigma in sigmas:
        for threshold in thresholds:
            # Generate labels with current parameters
            df_labeled = create_oracle_labels(df, sigma=int(sigma), threshold=threshold)
            
            # Calculate class distribution
            counts = df_labeled['target'].value_counts(normalize=True)
            
            results.append({
                'sigma': int(sigma),
                'threshold': threshold,
                'down_pct': counts.get(0, 0) * 100,
                'sideways_pct': counts.get(1, 0) * 100,
                'up_pct': counts.get(2, 0) * 100,
                'imbalance': abs(counts.get(0, 0) - counts.get(2, 0)) * 100
            })
    
    # Return sorted by imbalance (most balanced first)
    return pd.DataFrame(results).sort_values('imbalance')


def shift_target_for_prediction(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Shift target backward so current features predict future labels.
    
    After calling this function:
    - Features at time t correspond to target at time t+horizon
    - This is necessary for training a predictive model
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'target' column from create_oracle_labels().
    horizon : int, default=1
        How many bars ahead to predict.
        - horizon=1: predict next bar's direction
        - horizon=5: predict direction 5 bars ahead
    
    Returns
    -------
    pd.DataFrame
        DataFrame with shifted target. Rows without valid target are dropped.
    
    Notes
    -----
    The shift is NEGATIVE (backward) because we want:
    - Row at time t to have target from time t+horizon
    - This means: "using features from now, predict future direction"
    """
    df_result = df.copy()
    
    # Shift target backward (negative shift moves future values to current row)
    df_result['target'] = df_result['target'].shift(-horizon)
    
    # Drop rows where target is NaN (last 'horizon' rows)
    df_result = df_result.dropna(subset=['target'])
    
    # Ensure target is integer type
    df_result['target'] = df_result['target'].astype(int)
    
    return df_result


def get_label_name(label: int) -> str:
    """
    Convert numeric label to human-readable string.
    
    Parameters
    ----------
    label : int
        Numeric label (0, 1, or 2).
    
    Returns
    -------
    str
        'DOWN', 'SIDEWAYS', or 'UP'.
    """
    mapping = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
    return mapping.get(label, 'UNKNOWN')


def get_label_color(label: int) -> str:
    """
    Get visualization color for each label.
    
    Parameters
    ----------
    label : int
        Numeric label (0, 1, or 2).
    
    Returns
    -------
    str
        Hex color code suitable for matplotlib.
    """
    colors = {
        0: '#ff6b6b',  # Red for DOWN
        1: '#a0a0a0',  # Gray for SIDEWAYS
        2: '#51cf66'   # Green for UP
    }
    return colors.get(label, '#ffffff')


# =============================================================================
# Module Test / Demo
# =============================================================================
if __name__ == "__main__":
    """
    Test the oracle labeling with synthetic data.
    This runs when the module is executed directly.
    """
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing Oracle Labels Module")
    print("=" * 60)
    
    # Create synthetic price data that mimics BTC behavior
    np.random.seed(42)
    n = 1000
    
    # Simulate price with random walk + drift
    log_returns = np.random.randn(n) * 0.01  # 1% daily volatility
    log_prices = np.cumsum(log_returns)
    prices = 100 * np.exp(log_prices)  # Start at $100
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='15min'),
        'close': prices
    })
    
    # Generate oracle labels
    df_labeled = create_oracle_labels(df, sigma=4, threshold=0.0004)
    
    # Print distribution
    print("\nLabel Distribution:")
    print("-" * 30)
    for label in [0, 1, 2]:
        count = (df_labeled['target'] == label).sum()
        pct = count / len(df_labeled) * 100
        print(f"  {get_label_name(label):10s}: {count:5d} ({pct:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot 1: Price with smoothed oracle line
    axes[0].plot(df_labeled['time'], df_labeled['close'], 
                 label='Close Price', alpha=0.7, linewidth=0.8)
    axes[0].plot(df_labeled['time'], df_labeled['smoothed_close'], 
                 label='Oracle (smoothed)', linewidth=2, color='orange')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].set_title('Price with Oracle Smoothing Line')
    
    # Plot 2: Labels as colored background
    for label in [0, 1, 2]:
        mask = df_labeled['target'] == label
        axes[1].fill_between(
            df_labeled['time'], 0, 1,
            where=mask,
            alpha=0.6,
            color=get_label_color(label),
            label=get_label_name(label)
        )
    axes[1].set_ylabel('Labels')
    axes[1].legend()
    axes[1].set_title('Oracle Labels (DOWN=Red, SIDEWAYS=Gray, UP=Green)')
    
    plt.tight_layout()
    plt.savefig('oracle_labels_demo.png', dpi=150)
    print(f"\nVisualization saved to: oracle_labels_demo.png")
