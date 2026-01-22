"""
Feature Builder Module for Bitcoin Price Prediction.

This module orchestrates the feature engineering pipeline:
1. Calculate all technical indicators
2. Shift features for prediction (avoid look-ahead bias)
3. Handle missing values (NaN from indicator warmup)
4. Scale features for model training
5. Create sequences for LSTM models

Key Concepts:
-------------
- Features must be LAGGED to predict the future
- Scalers must be fit ONLY on training data
- Different window sizes create different amounts of NaN at start

Data Leakage Prevention:
-----------------------
1. NEVER use future data in features
2. FIT scaler on train data only, TRANSFORM val/test
3. Use time-based splits, not random splits
4. Shift target AFTER calculating indicators

Author: Capstone Project
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import warnings

from src.features.indicators import add_all_indicators, get_indicator_columns, GROUP_FUNCS

warnings.filterwarnings('ignore')


def prepare_features(df: pd.DataFrame,
                     groups: Optional[List[str]] = None,
                     horizon: int = 1,
                     dropna: bool = True) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Complete feature preparation pipeline.
    
    This is the main function for feature engineering. It:
    1. Adds technical indicators from specified groups
    2. Creates log returns (percentage change in log space)
    3. Shifts features for prediction horizon
    4. Optionally drops NaN values
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data AND target column (from create_oracle_labels).
    groups : List[str], optional
        Indicator groups to add. Default: all groups.
        Options: 'momentum', 'overlap', 'trend', 'volatility', 
                 'volume', 'statistics', 'candle'
    horizon : int, default=1
        Number of bars ahead to predict.
        Features at time t should predict target at time t+horizon.
    dropna : bool, default=True
        Whether to drop rows with NaN values.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, List[str]]]
        - DataFrame with features and shifted target
        - Dictionary mapping group names to their column names
    
    Examples
    --------
    >>> df_labeled = create_oracle_labels(df)
    >>> df_features, group_map = prepare_features(df_labeled, horizon=1)
    >>> print(f"Total features: {sum(len(v) for v in group_map.values())}")
    """
    print("\n" + "=" * 60)
    print("🔧 PREPARING FEATURES")
    print("=" * 60)
    
    df_result = df.copy()
    
    # Use all groups if not specified
    if groups is None:
        groups = list(GROUP_FUNCS.keys())
    
    print(f"Groups to add: {groups}")
    
    # Step 1: Add technical indicators
    print("\n📊 Adding indicators...")
    
    # Track columns before to identify new ones
    group_columns: Dict[str, List[str]] = {}
    
    for group in groups:
        if group in GROUP_FUNCS:
            cols_before = set(df_result.columns)
            print(f"  -> Processing group: {group}")
            
            # Apply group function
            df_result = GROUP_FUNCS[group](df_result)
            
            # Identify added columns
            cols_after = set(df_result.columns)
            new_cols = list(cols_after - cols_before)
            
            # Store in map
            group_columns.setdefault(group, []).extend(new_cols)
            
            print(f"     Added {len(new_cols)} features: {', '.join(new_cols[:5])}..." if len(new_cols) > 5 else f"     Added {len(new_cols)} features: {', '.join(new_cols)}")
        else:
            print(f"  ⚠️ Warning: Unknown group '{group}'")

    # Step 2: Add log returns (useful feature)
    if 'log_return' not in df_result.columns:
        df_result['log_return'] = np.log(df_result['close'] / df_result['close'].shift(1))
    
    # Step 3: Categorize any remaining columns (if any were added outside the loop)
    # This logic is now simpler because we tracked them in the loop above
    all_indicator_cols = get_indicator_columns(df_result)
    for col in all_indicator_cols:
        found = False
        for g in group_columns:
            if col in group_columns[g]:
                found = True
                break
        if not found:
            group_columns.setdefault('other', []).append(col)
    
    # Step 4: Shift target for prediction horizon
    # This ensures we predict FUTURE direction, not current
    if 'target' in df_result.columns and horizon > 0:
        print(f"\n⏰ Shifting target by -{horizon} for prediction...")
        # Shift target backward (negative = future values move to current row)
        df_result['target'] = df_result['target'].shift(-horizon)
    
    # Step 5: Handle inf and NaN values
    rows_before = len(df_result)
    if dropna:
        # Replace inf with NaN (some indicators produce inf from division by zero)
        df_result = df_result.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill funding rate columns (they update every 8 hours, not every 15 min)
        funding_cols = ['funding_interval_hours', 'last_funding_rate']
        for col in funding_cols:
            if col in df_result.columns:
                df_result[col] = df_result[col].ffill().bfill()
        
        # Find the maximum warmup period (max NaN count at the start of any column)
        # This is how many rows we need to drop from the beginning
        nan_counts = df_result.isna().sum()
        
        # Show top NaN columns for debugging
        top_nan = nan_counts.sort_values(ascending=False).head(5)
        if top_nan.max() > 0:
            print(f"\n📋 Top NaN columns:")
            for col, cnt in top_nan.items():
                print(f"   {col}: {cnt} NaN")
        
        max_warmup = int(nan_counts.max()) if len(nan_counts) > 0 else 0
        
        # Cap max_warmup at 300 rows to avoid losing too much data
        # If an indicator needs more warmup, we'll fill with 0 instead
        WARMUP_CAP = 300
        if max_warmup > WARMUP_CAP:
            print(f"\n⚠️ Max warmup {max_warmup} exceeds cap {WARMUP_CAP}, using cap")
            max_warmup = WARMUP_CAP
        
        # Also account for the last row NaN from target shift
        # Drop first max_warmup rows and last 1 row
        if max_warmup > 0 or 'target' in df_result.columns:
            drop_start = max_warmup
            drop_end = 1 if 'target' in df_result.columns else 0
            df_result = df_result.iloc[drop_start:-drop_end] if drop_end > 0 else df_result.iloc[drop_start:]
            print(f"\n📋 Warmup period: {drop_start} rows, target shift: {drop_end} row")
        
        # Fill any remaining NaN with 0 (rare edge cases)
        remaining_nan = df_result.isna().sum().sum()
        if remaining_nan > 0:
            print(f"   Filling {remaining_nan} remaining NaN with 0")
            df_result = df_result.fillna(0)
        
        # Fill any remaining NaN with 0 (rare edge cases)
        remaining_nan = df_result.isna().sum().sum()
        if remaining_nan > 0:
            print(f"   Filling {remaining_nan} remaining NaN with 0")
            df_result = df_result.fillna(0)
        
    rows_after = len(df_result)
    print(f"\n🧹 Dropped {rows_before - rows_after} rows with NaN ({rows_after:,} remaining)")
    
    # Check if we dropped too many rows
    if rows_after == 0:
        print("\n⚠️ WARNING: All rows were dropped! Check if:")
        print("   1. Oracle labels were created (target column exists)")
        print("   2. Data has enough rows for indicator warmup (~50)")
        print("   3. Some columns are entirely NaN")
    
    # Step 6: Convert target to integer (only if dropna was True, so no NaN remain)
    if 'target' in df_result.columns and len(df_result) > 0 and dropna:
        df_result['target'] = df_result['target'].astype(int)
    
    # Summary
    print(f"\n📊 Feature Summary:")
    print(f"   Total rows: {len(df_result):,}")
    print(f"   Total columns: {len(df_result.columns)}")
    print(f"   Indicator groups:")
    for group, cols in group_columns.items():
        print(f"      {group}: {len(cols)} features")
    
    return df_result, group_columns


def get_feature_matrix(df: pd.DataFrame,
                       feature_cols: Optional[List[str]] = None,
                       target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    feature_cols : List[str], optional
        List of feature column names. If None, auto-detect.
    target_col : str, default='target'
        Name of target column.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X (n_samples, n_features), y (n_samples,)
    """
    if feature_cols is None:
        feature_cols = get_indicator_columns(df)
    
    # Filter to only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int32)
    
    return X, y


def select_top_k_features(df: pd.DataFrame,
                          feature_importance: Dict[str, float],
                          k: int = 30) -> List[str]:
    """
    Select top K features by importance.
    
    Use this after training to reduce feature set for faster inference.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features.
    feature_importance : Dict[str, float]
        Dictionary mapping feature names to importance scores.
        Higher score = more important.
    k : int, default=30
        Number of top features to select.
    
    Returns
    -------
    List[str]
        Names of top K features.
    """
    # Get indicator columns that exist in both df and importance dict
    available_features = [c for c in get_indicator_columns(df) 
                         if c in feature_importance]
    
    # Sort by importance  
    sorted_features = sorted(available_features, 
                            key=lambda x: feature_importance.get(x, 0),
                            reverse=True)
    
    return sorted_features[:k]


class FeatureScaler:
    """
    Feature scaler with save/load functionality.
    
    This class wraps sklearn's RobustScaler and adds:
    - Easy save/load to disk
    - Column name tracking
    - Fit status tracking
    
    RobustScaler is preferred over StandardScaler because:
    - Uses median and IQR instead of mean and std
    - Less sensitive to outliers (common in financial data)
    
    Usage Example:
    -------------
    >>> scaler = FeatureScaler()
    >>> X_train_scaled = scaler.fit_transform(X_train, feature_names)
    >>> X_test_scaled = scaler.transform(X_test)
    >>> scaler.save('models_artifacts/scaler.joblib')
    
    >>> # Later...
    >>> scaler = FeatureScaler.load('models_artifacts/scaler.joblib')
    >>> X_new_scaled = scaler.transform(X_new)
    """
    
    def __init__(self):
        """Initialize empty scaler."""
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeatureScaler':
        """
        Fit scaler on training data.
        
        IMPORTANT: Only call this on TRAINING data!
        Using validation or test data would cause data leakage.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix (n_samples, n_features).
        feature_names : List[str], optional
            Names of feature columns for reference.
        
        Returns
        -------
        self
        """
        self.scaler.fit(X)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform.
        
        Returns
        -------
        np.ndarray
            Scaled feature matrix.
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, 
                      feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit scaler and transform in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix.
        feature_names : List[str], optional
            Names of feature columns.
        
        Returns
        -------
        np.ndarray
            Scaled feature matrix.
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def save(self, filepath: str) -> None:
        """
        Save scaler to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the scaler (e.g., 'models/scaler.joblib').
        """
        save_dict = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, filepath)
        print(f"✅ Scaler saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureScaler':
        """
        Load scaler from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved scaler.
        
        Returns
        -------
        FeatureScaler
            Loaded scaler instance.
        """
        save_dict = joblib.load(filepath)
        instance = cls()
        instance.scaler = save_dict['scaler']
        instance.feature_names = save_dict['feature_names']
        instance.is_fitted = save_dict['is_fitted']
        print(f"✅ Scaler loaded from {filepath}")
        return instance


def create_sequences_for_lstm(X: np.ndarray, 
                              y: np.ndarray, 
                              lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D sequences for LSTM/CNN-LSTM models.
    
    Converts 2D tabular data (samples, features) into 3D sequences
    (samples, timesteps, features) required by LSTM layers.
    
    After this transformation:
    - Each sample contains 'lookback' previous timesteps
    - The target corresponds to the LAST timestep in each sequence
    
    Parameters
    ----------
    X : np.ndarray
        2D feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        1D target array of shape (n_samples,).
    lookback : int, default=20
        Number of previous timesteps to include in each sequence.
        Higher lookback = more historical context, but more memory usage.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_seq : 3D array of shape (n_sequences, lookback, n_features)
        y_seq : 1D array of shape (n_sequences,)
    
    Notes
    -----
    The number of output samples is: n_samples - lookback
    The first 'lookback' samples cannot be used as they don't have
    enough history to form a complete sequence.
    
    Examples
    --------
    >>> X = np.random.randn(1000, 30)  # 1000 samples, 30 features
    >>> y = np.random.randint(0, 3, 1000)
    >>> X_seq, y_seq = create_sequences_for_lstm(X, y, lookback=20)
    >>> print(X_seq.shape)  # (980, 20, 30)
    >>> print(y_seq.shape)  # (980,)
    """
    n_samples, n_features = X.shape
    n_sequences = n_samples - lookback
    
    # Pre-allocate arrays for efficiency
    X_seq = np.zeros((n_sequences, lookback, n_features), dtype=np.float32)
    y_seq = np.zeros(n_sequences, dtype=np.int32)
    
    # Create sequences by sliding window
    for i in range(n_sequences):
        # Features: lookback previous timesteps
        X_seq[i] = X[i:i + lookback]
        # Target: label at the current (last) timestep
        y_seq[i] = y[i + lookback]
    
    return X_seq, y_seq


# =============================================================================
# Module Test
# =============================================================================
if __name__ == "__main__":
    """Test feature builder with synthetic data."""
    
    print("=" * 60)
    print("Testing Feature Builder Module")
    print("=" * 60)
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 500
    
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='15min'),
        'open': close_prices + np.random.randn(n) * 0.3,
        'high': close_prices + np.abs(np.random.randn(n) * 0.5),
        'low': close_prices - np.abs(np.random.randn(n) * 0.5),
        'close': close_prices,
        'volume': np.random.rand(n) * 1000 + 100,
        'target': np.random.randint(0, 3, n)  # Dummy target
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    print(f"\nInput shape: {df.shape}")
    
    # Test prepare_features
    df_features, group_map = prepare_features(df, horizon=1)
    
    print(f"\nOutput shape: {df_features.shape}")
    
    # Test feature matrix extraction
    feature_cols = get_indicator_columns(df_features)
    X, y = get_feature_matrix(df_features, feature_cols)
    
    print(f"\nFeature matrix X: {X.shape}")
    print(f"Target vector y: {y.shape}")
    
    # Test scaler
    print("\nTesting scaler...")
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X, feature_cols)
    print(f"Scaled X mean: {X_scaled.mean():.4f}")
    print(f"Scaled X std: {X_scaled.std():.4f}")
    
    # Test LSTM sequence creation
    print("\nTesting LSTM sequence creation...")
    X_seq, y_seq = create_sequences_for_lstm(X_scaled, y, lookback=20)
    print(f"Sequence X shape: {X_seq.shape}")
    print(f"Sequence y shape: {y_seq.shape}")
    
    print("\n✅ Feature builder module test complete!")
