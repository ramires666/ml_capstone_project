"""
Compare model performance across different prediction horizons.

Usage:
    python scripts/compare_horizons.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import json
from datetime import datetime

from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.xgb import XGBBaseline
from src.models.cnn_lstm import CNNLSTMModel
from src.models.train import train_baseline, train_cnn_lstm, load_config


def compare_horizons(horizons=[1, 3, 5], train_models=True):
    """Train and compare models for different horizons."""
    
    print("="*60)
    print("🔄 HORIZON COMPARISON")
    print("="*60)
    
    results = []
    
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"📊 HORIZON = {horizon} bar(s)")
        print("="*60)
        
        if train_models:
            # Train XGBoost
            baseline_config = load_config('configs/baseline.yaml')
            baseline_config['features']['horizon'] = horizon
            xgb_metrics = train_baseline(baseline_config, horizon=horizon)
            
            results.append({
                'Horizon': horizon,
                'Model': 'XGBoost',
                'Accuracy': xgb_metrics['accuracy'],
                'F1 Weighted': xgb_metrics['f1_weighted'],
                'F1 Macro': xgb_metrics['f1_macro']
            })
            
            # Train CNN-LSTM
            cnn_config = load_config('configs/cnn_lstm.yaml')
            cnn_config['features']['horizon'] = horizon
            cnn_metrics = train_cnn_lstm(cnn_config, horizon=horizon)
            
            results.append({
                'Horizon': horizon,
                'Model': 'CNN-LSTM',
                'Accuracy': cnn_metrics['accuracy'],
                'F1 Weighted': cnn_metrics['f1_weighted'],
                'F1 Macro': cnn_metrics['f1_macro']
            })
        else:
            # Just load and evaluate existing models
            try:
                xgb = XGBBaseline.load('models_artifacts', name=f'xgb_baseline_h{horizon}')
                results.append({
                    'Horizon': horizon,
                    'Model': 'XGBoost',
                    **xgb.metrics
                })
            except:
                print(f"XGBoost h={horizon} not found")
            
            try:
                cnn = CNNLSTMModel.load('models_artifacts', name=f'cnn_lstm_h{horizon}')
                results.append({
                    'Horizon': horizon,
                    'Model': 'CNN-LSTM',
                    **cnn.metrics
                })
            except:
                print(f"CNN-LSTM h={horizon} not found")
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("📊 FINAL COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    output_dir = Path('reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / 'horizon_comparison.csv', index=False)
    
    with open(output_dir / 'horizon_comparison.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'horizons': horizons,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}/")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 5])
    parser.add_argument('--no-train', action='store_true', help='Only evaluate existing models')
    
    args = parser.parse_args()
    
    compare_horizons(horizons=args.horizons, train_models=not args.no_train)
