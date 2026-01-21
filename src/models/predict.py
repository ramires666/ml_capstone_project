"""
Prediction Script for Bitcoin Price Direction

Usage:
    python -m src.models.predict --model xgb --horizon 1 --data data.csv
    python -m src.models.predict --model cnn_lstm --horizon 1 --latest
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_and_merge_data
from src.features.builder import prepare_features
from src.features.indicators import add_all_indicators, get_indicator_columns
from src.models.xgb import XGBBaseline
from src.models.cnn_lstm import CNNLSTMModel


LABEL_NAMES = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
LABEL_COLORS = {0: '🔴', 1: '⚪', 2: '🟢'}


class Predictor:
    """
    Unified predictor for both XGBoost and CNN-LSTM models.
    """
    
    def __init__(self, 
                 model_type: str = 'xgb',
                 model_dir: str = 'models_artifacts',
                 horizon: int = 1,
                 device: str = 'cuda'):
        """
        Initialize predictor.
        
        Args:
            model_type: 'xgb' or 'cnn_lstm'
            model_dir: Directory with saved models
            horizon: Prediction horizon
            device: Device for CNN-LSTM ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.horizon = horizon
        self.device = device
        
        self.model = None
        self.feature_names = []
        
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model."""
        name = f'{self.model_type}_baseline_h{self.horizon}' if self.model_type == 'xgb' \
               else f'cnn_lstm_h{self.horizon}'
        
        if self.model_type == 'xgb':
            self.model = XGBBaseline.load(self.model_dir, name=name)
        else:
            self.model = CNNLSTMModel.load(self.model_dir, name=name, device=self.device)
        
        self.feature_names = self.model.feature_names
        print(f"✅ Loaded {self.model_type} model (horizon={self.horizon})")
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for prediction."""
        # Add all indicators
        df = add_all_indicators(df)
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Make prediction for the latest data point.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dict with prediction, probabilities, timestamp
        """
        # Preprocess
        df_processed = self.preprocess(df)
        
        if df_processed.empty:
            return {'error': 'Not enough data after preprocessing'}
        
        # Get features
        available_features = [c for c in self.feature_names if c in df_processed.columns]
        
        if len(available_features) < len(self.feature_names) * 0.5:
            return {'error': f'Missing too many features. Expected {len(self.feature_names)}, got {len(available_features)}'}
        
        X = df_processed[available_features].values
        
        # Predict probabilities
        proba = self.model.predict_proba(X)
        
        # Get latest prediction
        latest_proba = proba[-1]
        latest_pred = int(np.argmax(latest_proba))
        
        # Get timestamp
        if 'time' in df_processed.columns:
            timestamp = str(df_processed['time'].iloc[-1])
        else:
            timestamp = str(df_processed.index[-1])
        
        return {
            'prediction': latest_pred,
            'label': LABEL_NAMES[latest_pred],
            'icon': LABEL_COLORS[latest_pred],
            'probabilities': {
                'DOWN': float(latest_proba[0]),
                'SIDEWAYS': float(latest_proba[1]) if len(latest_proba) > 2 else 0,
                'UP': float(latest_proba[-1])
            },
            'timestamp': timestamp,
            'model': self.model_type,
            'horizon': self.horizon
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all rows.
        
        Returns:
            DataFrame with predictions added
        """
        df_processed = self.preprocess(df)
        
        available_features = [c for c in self.feature_names if c in df_processed.columns]
        X = df_processed[available_features].values
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Create result DataFrame
        result = df_processed.copy()
        
        # Align predictions with original indices
        if self.model_type == 'cnn_lstm':
            # CNN-LSTM drops lookback rows
            lookback = self.model.lookback
            result = result.iloc[lookback:].copy()
        
        result['prediction'] = predictions
        result['prob_down'] = probabilities[:, 0]
        result['prob_sideways'] = probabilities[:, 1] if probabilities.shape[1] > 2 else 0
        result['prob_up'] = probabilities[:, -1]
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained models')
    parser.add_argument('--model', type=str, choices=['xgb', 'cnn_lstm'], default='xgb')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--model-dir', type=str, default='models_artifacts')
    parser.add_argument('--data', type=str, help='Path to CSV/Parquet with OHLCV data')
    parser.add_argument('--latest', action='store_true', help='Use latest data from loader')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load data
    if args.data:
        if args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)
    elif args.latest:
        df = load_and_merge_data()
    else:
        print("Error: Provide --data or --latest")
        return
    
    # Create predictor
    predictor = Predictor(
        model_type=args.model,
        model_dir=args.model_dir,
        horizon=args.horizon,
        device=args.device
    )
    
    # Make prediction
    result = predictor.predict(df)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Print result
    print("\n" + "="*50)
    print(f"🔮 Prediction for next {args.horizon} bar(s)")
    print("="*50)
    print(f"\n{result['icon']} Direction: {result['label']}")
    print(f"\nProbabilities:")
    print(f"   DOWN:     {result['probabilities']['DOWN']*100:.1f}%")
    print(f"   SIDEWAYS: {result['probabilities']['SIDEWAYS']*100:.1f}%")
    print(f"   UP:       {result['probabilities']['UP']*100:.1f}%")
    print(f"\nTimestamp: {result['timestamp']}")
    print(f"Model: {result['model']} (horizon={result['horizon']})")


if __name__ == "__main__":
    main()
