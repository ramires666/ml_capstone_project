"""
Feature Builder for Bitcoin Price Prediction

Builds complete feature set from raw data:
1. Adds technical indicators
2. Shifts features for prediction horizon
3. Handles NaN values
4. Optionally scales features

Based on old_project_files/train.py prepare_all_features() with improvements.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Dict, Optional, Tuple
import joblib
from pathlib import Path

from .indicators import add_all_indicators, get_indicator_columns, GROUP_FUNCS


# Columns that should never be used as features
EXCLUDED_COLS = [
    'time', 'target', 'smoothed_close', 'smooth_slope',
    'open_time', 'close_time'
]


def prepare_features(df: pd.DataFrame,
                     groups: Optional[List[str]] = None,
                     horizon: int = 1,
                     drop_na: bool = True) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Prepare all features for training/inference.
    
    Pipeline:
    1. Calculate all indicator groups
    2. Map features to groups for analysis
    3. Shift all features by horizon to prevent lookahead bias
    4. Drop NaN rows
    
    Args:
        df: DataFrame with OHLCV + target
        groups: List of indicator groups to add (None = all)
        horizon: Prediction horizon in bars
        drop_na: Whether to drop rows with NaN
    
    Returns:
        Tuple of:
        - DataFrame with features ready for training
        - Dict mapping group names to their feature columns
    """
    print(f"\n🔧 Preparing features (horizon={horizon})...")
    
    df_result = df.copy()
    
    # Ensure time is proper datetime
    if 'time' in df_result.columns:
        df_result['time'] = pd.to_datetime(df_result['time'])
    
    # Store columns before adding indicators
    base_cols = set(df_result.columns)
    
    # Add indicators
    if groups is None:
        groups = list(GROUP_FUNCS.keys())
    
    group_features_map = {}
    
    for group_name in groups:
        if group_name not in GROUP_FUNCS:
            print(f"⚠️ Unknown group: {group_name}")
            continue
        
        cols_before = set(df_result.columns)
        
        try:
            df_result = GROUP_FUNCS[group_name](df_result)
        except Exception as e:
            print(f"⚠️ Error in group '{group_name}': {e}")
            continue
        
        cols_after = set(df_result.columns)
        new_cols = list(cols_after - cols_before)
        group_features_map[group_name] = new_cols
        
        print(f"   {group_name}: +{len(new_cols)} features")
    
    # Drop columns that are 100% NaN
    nan_cols = df_result.columns[df_result.isna().all()].tolist()
    if nan_cols:
        print(f"   Dropping {len(nan_cols)} all-NaN columns")
        df_result = df_result.drop(columns=nan_cols)
        
        # Update group map
        for g in group_features_map:
            group_features_map[g] = [c for c in group_features_map[g] if c not in nan_cols]
    
    # Get feature columns
    feature_cols = get_indicator_columns(df_result, EXCLUDED_COLS)
    
    # Add base features (OHLCV ratios, log returns, etc.)
    if 'log_return' not in df_result.columns and 'close' in df_result.columns:
        df_result['log_return'] = np.log(df_result['close'] / df_result['close'].shift(1))
        feature_cols.append('log_return')
    
    # Shift all features by horizon to prevent lookahead bias
    # This is CRITICAL for valid backtesting!
    print(f"   Shifting features by {horizon} bars...")
    
    for col in feature_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].shift(horizon)
    
    # Drop NaN rows
    if drop_na:
        before_len = len(df_result)
        df_result = df_result.dropna(subset=feature_cols + ['target'])
        after_len = len(df_result)
        print(f"   Dropped {before_len - after_len} NaN rows")
    
    print(f"✅ Features ready: {len(feature_cols)} features, {len(df_result)} rows")
    
    return df_result, group_features_map


def get_feature_matrix(df: pd.DataFrame,
                       feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target y from DataFrame.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature columns to use (None = auto-detect)
    
    Returns:
        Tuple of (X, y) numpy arrays
    """
    if feature_cols is None:
        feature_cols = get_indicator_columns(df, EXCLUDED_COLS)
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].values
    y = df['target'].values.astype(int)
    
    return X, y


def select_top_features(df: pd.DataFrame,
                        feature_importances: pd.DataFrame,
                        top_k: int = 30) -> List[str]:
    """
    Select top K features by importance.
    
    Args:
        df: DataFrame with features
        feature_importances: DataFrame with 'Feature' and 'Importance' columns
        top_k: Number of top features to select
    
    Returns:
        List of top K feature names
    """
    # Sort by importance
    fi_sorted = feature_importances.sort_values('Importance', ascending=False)
    
    # Get top K that exist in df
    top_features = []
    for feat in fi_sorted['Feature']:
        if feat in df.columns and feat not in EXCLUDED_COLS:
            top_features.append(feat)
            if len(top_features) >= top_k:
                break
    
    return top_features


class FeatureScaler:
    """
    Scaler wrapper for features with save/load functionality.
    
    Uses RobustScaler by default (handles outliers better for financial data).
    """
    
    def __init__(self, scaler_type: str = 'robust'):
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> 'FeatureScaler':
        """Fit scaler on training data."""
        if feature_cols is None:
            feature_cols = get_indicator_columns(df, EXCLUDED_COLS)
        
        self.feature_names = [c for c in feature_cols if c in df.columns]
        X = df[self.feature_names].values
        
        self.scaler.fit(X)
        self.is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        df_result = df.copy()
        
        # Get features that exist in both scaler and df
        available_features = [c for c in self.feature_names if c in df_result.columns]
        
        if available_features:
            X = df_result[available_features].values
            X_scaled = self.scaler.transform(X)
            df_result[available_features] = X_scaled
        
        return df_result
    
    def fit_transform(self, df: pd.DataFrame, 
                      feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, feature_cols)
        return self.transform(df)
    
    def save(self, path: str):
        """Save scaler to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        """Load scaler from file."""
        data = joblib.load(path)
        
        instance = cls()
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.is_fitted = True
        
        return instance


def create_sequences_for_lstm(X: np.ndarray,
                               y: np.ndarray,
                               lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM from tabular data.
    
    Transforms (samples, features) → (samples, lookback, features)
    
    Args:
        X: Feature matrix (samples, features)
        y: Target array (samples,)
        lookback: Sequence length
    
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    n_samples = len(X) - lookback
    n_features = X.shape[1]
    
    X_seq = np.zeros((n_samples, lookback, n_features))
    y_seq = np.zeros(n_samples)
    
    for i in range(n_samples):
        X_seq[i] = X[i:i+lookback]
        y_seq[i] = y[i+lookback]
    
    return X_seq, y_seq


if __name__ == "__main__":
    # Test
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.data.loader import load_and_merge_data
    from src.labeling.oracle import create_oracle_labels
    
    print("Loading data...")
    # This will fail without actual data, just for testing structure
    try:
        df = load_and_merge_data()
        df = create_oracle_labels(df)
        
        df_features, group_map = prepare_features(df, horizon=1)
        
        print("\nGroup features:")
        for group, cols in group_map.items():
            print(f"  {group}: {len(cols)} features")
        
    except FileNotFoundError:
        print("No data files found. Run downloader and parser first.")
