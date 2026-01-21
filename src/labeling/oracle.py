"""
Oracle Labels Generator

Creates target labels using Gaussian smoothing (Oracle approach).
Based on old_project_files/train.py create_target_labels() with improvements.

The "oracle" approach uses centered Gaussian smoothing to identify the true
underlying trend, then classifies each point as UP/DOWN/SIDEWAYS based on the slope.

IMPORTANT: The oracle line is used ONLY for generating labels, not as a feature!
           Using it as a feature would cause look-ahead bias (leakage).
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional
from numba import jit


@jit(nopython=True, cache=True)
def classify_slope_numba(slope: np.ndarray, threshold: float) -> np.ndarray:
    """
    Numba-accelerated classification of slope values.
    
    Args:
        slope: Array of slope values
        threshold: Threshold for UP/DOWN classification
    
    Returns:
        Array of labels: 0=DOWN, 1=SIDEWAYS, 2=UP
    """
    n = len(slope)
    labels = np.ones(n, dtype=np.int32)  # Default: SIDEWAYS
    
    for i in range(n):
        if slope[i] > threshold:
            labels[i] = 2  # UP
        elif slope[i] < -threshold:
            labels[i] = 0  # DOWN
    
    return labels


def create_oracle_labels(df: pd.DataFrame,
                         sigma: int = 4,
                         threshold: float = 0.0004,
                         price_col: str = 'close') -> pd.DataFrame:
    """
    Create target labels using Gaussian smoothing (Oracle approach).
    
    The algorithm:
    1. Apply centered Gaussian smoothing to log(close) prices
    2. Calculate the slope (log-difference) of the smoothed line
    3. Classify based on threshold:
       - slope > threshold  → UP (2)
       - slope < -threshold → DOWN (0)
       - otherwise          → SIDEWAYS (1)
    
    Args:
        df: DataFrame with OHLC data
        sigma: Gaussian kernel standard deviation (higher = more smoothing)
        threshold: Slope threshold for UP/DOWN classification
        price_col: Column name for price (default: 'close')
    
    Returns:
        DataFrame with added columns:
        - target: int (0=DOWN, 1=SIDEWAYS, 2=UP)
        - smoothed_close: Smoothed price (for visualization only!)
        - smooth_slope: Slope value (for visualization only!)
    
    WARNING: smoothed_close and smooth_slope are for analysis only!
             Do NOT use them as features - they cause look-ahead bias!
    """
    df_result = df.copy()
    
    # Get close prices
    close = df_result[price_col].values.astype(np.float64)
    
    # Apply Gaussian smoothing to log prices
    log_close = np.log(close)
    smoothed_log = gaussian_filter1d(log_close, sigma=sigma, mode='nearest')
    
    # Convert back to price space for visualization
    df_result['smoothed_close'] = np.exp(smoothed_log)
    
    # Calculate slope (log-difference of smoothed line)
    smooth_slope = np.diff(smoothed_log, prepend=smoothed_log[0])
    df_result['smooth_slope'] = smooth_slope
    
    # Classify with numba acceleration
    labels = classify_slope_numba(smooth_slope, threshold)
    df_result['target'] = labels
    
    return df_result


def analyze_label_distribution(df: pd.DataFrame,
                               sigma_range: Tuple[int, int] = (1, 10),
                               threshold_range: Tuple[float, float] = (0.0001, 0.001),
                               n_steps: int = 5) -> pd.DataFrame:
    """
    Analyze target distribution for different sigma and threshold values.
    
    Useful for finding parameters that give balanced class distribution.
    
    Args:
        df: DataFrame with OHLC data
        sigma_range: Range of sigma values to test (min, max)
        threshold_range: Range of threshold values to test (min, max)
        n_steps: Number of steps for each parameter
    
    Returns:
        DataFrame with columns: sigma, threshold, down_pct, sideways_pct, up_pct
    """
    results = []
    
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_steps, dtype=int)
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    
    for sigma in sigmas:
        for threshold in thresholds:
            df_labeled = create_oracle_labels(df, sigma=int(sigma), threshold=threshold)
            
            counts = df_labeled['target'].value_counts(normalize=True)
            
            results.append({
                'sigma': int(sigma),
                'threshold': threshold,
                'down_pct': counts.get(0, 0) * 100,
                'sideways_pct': counts.get(1, 0) * 100,
                'up_pct': counts.get(2, 0) * 100,
                'imbalance': abs(counts.get(0, 0) - counts.get(2, 0)) * 100
            })
    
    return pd.DataFrame(results).sort_values('imbalance')


def shift_target_for_prediction(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Shift target backward so we predict future labels.
    
    After this operation:
    - Features at time t
    - Target is the label at time t+horizon
    
    Args:
        df: DataFrame with 'target' column
        horizon: How many bars ahead to predict
    
    Returns:
        DataFrame with shifted target (rows with NaN target are dropped)
    """
    df_result = df.copy()
    
    # Shift target backward (so current row predicts future)
    df_result['target'] = df_result['target'].shift(-horizon)
    
    # Drop rows without target
    df_result = df_result.dropna(subset=['target'])
    df_result['target'] = df_result['target'].astype(int)
    
    return df_result


def get_label_name(label: int) -> str:
    """Convert numeric label to string name."""
    mapping = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
    return mapping.get(label, 'UNKNOWN')


def get_label_color(label: int) -> str:
    """Get color for visualization."""
    colors = {0: '#ff6b6b', 1: '#a0a0a0', 2: '#51cf66'}
    return colors.get(label, '#ffffff')


if __name__ == "__main__":
    # Test with dummy data
    import matplotlib.pyplot as plt
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    t = np.arange(n)
    
    # Simulate price with trend + noise
    trend = np.cumsum(np.random.randn(n) * 0.01)
    price = 100 * np.exp(trend)
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='15min'),
        'close': price
    })
    
    # Create labels
    df_labeled = create_oracle_labels(df, sigma=4, threshold=0.0004)
    
    # Print distribution
    print("Label distribution:")
    print(df_labeled['target'].value_counts(normalize=True).sort_index())
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Price and smoothed
    axes[0].plot(df_labeled['time'], df_labeled['close'], label='Close', alpha=0.7)
    axes[0].plot(df_labeled['time'], df_labeled['smoothed_close'], label='Oracle (smoothed)', lw=2)
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].set_title('Price with Oracle Line')
    
    # Labels as background
    for label in [0, 1, 2]:
        mask = df_labeled['target'] == label
        axes[1].fill_between(
            df_labeled['time'], 0, 1,
            where=mask,
            alpha=0.5,
            color=get_label_color(label),
            label=get_label_name(label)
        )
    axes[1].set_ylabel('Labels')
    axes[1].legend()
    axes[1].set_title('Oracle Labels')
    
    plt.tight_layout()
    plt.savefig('oracle_labels_test.png')
    print("\nSaved plot to oracle_labels_test.png")
