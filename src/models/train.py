"""
Training Script for Bitcoin Price Prediction Models

Usage:
    python -m src.models.train --config configs/baseline.yaml
    python -m src.models.train --config configs/cnn_lstm.yaml
    python -m src.models.train --config configs/baseline.yaml --horizon 3
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels, shift_target_for_prediction
from src.features.builder import prepare_features, get_feature_matrix
from src.features.indicators import get_indicator_columns
from src.models.xgb import XGBBaseline, print_classification_report
from src.models.cnn_lstm import CNNLSTMModel


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_baseline(config: dict, horizon: int = 1) -> dict:
    """
    Train XGBoost baseline model.
    
    Args:
        config: Configuration dict
        horizon: Prediction horizon in bars
    
    Returns:
        Metrics dict
    """
    print("\n" + "="*60)
    print("🚀 Training XGBoost Baseline Model")
    print("="*60)
    
    # Load data
    df = load_and_merge_data(
        start_date=config['data'].get('train_start'),
        end_date=config['data'].get('test_end')
    )
    
    # Create oracle labels
    df = create_oracle_labels(
        df,
        sigma=config['labeling']['sigma'],
        threshold=config['labeling']['threshold']
    )
    
    # Print label distribution
    print("\n📊 Label distribution:")
    print(df['target'].value_counts(normalize=True).sort_index())
    
    # Prepare features
    feature_groups = config['features'].get('groups')
    df_features, group_map = prepare_features(df, groups=feature_groups, horizon=horizon)
    
    # Split by time
    train_df, val_df, test_df = split_data_by_time(
        df_features,
        train_end=config['data']['train_end'],
        test_start=config['data']['test_start'],
        val_ratio=0.1
    )
    
    # Get feature columns
    feature_cols = get_indicator_columns(df_features, 
                                         ['time', 'target', 'smoothed_close', 'smooth_slope'])
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    print(f"\n📊 Features: {len(feature_cols)}")
    
    # Get X, y
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values.astype(int)
    
    X_val = val_df[feature_cols].values if len(val_df) > 0 else None
    y_val = val_df['target'].values.astype(int) if len(val_df) > 0 else None
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values.astype(int)
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape if X_val is not None else 'None'}")
    print(f"   Test:  {X_test.shape}")
    
    # Create and train model
    model = XGBBaseline(
        n_classes=3,
        device=config['xgboost'].get('device', 'cuda'),
        random_state=config['xgboost'].get('random_state', 42)
    )
    
    # Tune hyperparameters
    if config['training'].get('n_iter_search', 0) > 0:
        model.tune(
            X_train, y_train,
            param_dist=config.get('hyperparam_search'),
            n_iter=config['training']['n_iter_search'],
            cv_splits=config['training']['cv_splits'],
            scoring=config['training']['scoring']
        )
    else:
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    
    # Evaluate on test
    print("\n" + "="*60)
    print("📊 Test Set Evaluation")
    print("="*60)
    
    metrics = model.evaluate(X_test, y_test)
    print_classification_report(metrics, "XGBoost Baseline Results")
    
    # Feature importance
    fi = model.get_feature_importance()
    print("\n🔝 Top 10 Features:")
    print(fi.head(10).to_string(index=False))
    
    # Save model
    output_dir = Path(config['output']['model_dir'])
    model.save(output_dir, name=f'xgb_baseline_h{horizon}')
    
    # Save feature importance
    fi.to_csv(output_dir / f'feature_importance_h{horizon}.csv', index=False)
    
    # Add horizon to metrics
    metrics['horizon'] = horizon
    metrics['model_type'] = 'xgboost'
    
    return metrics


def train_cnn_lstm(config: dict, horizon: int = 1) -> dict:
    """
    Train CNN-LSTM model.
    
    Args:
        config: Configuration dict
        horizon: Prediction horizon in bars
    
    Returns:
        Metrics dict
    """
    print("\n" + "="*60)
    print("🚀 Training CNN-LSTM Model")
    print("="*60)
    
    # Load data
    df = load_and_merge_data(
        start_date=config['data'].get('train_start'),
        end_date=config['data'].get('test_end')
    )
    
    # Create oracle labels
    df = create_oracle_labels(
        df,
        sigma=config['labeling']['sigma'],
        threshold=config['labeling']['threshold']
    )
    
    # Print label distribution
    print("\n📊 Label distribution:")
    print(df['target'].value_counts(normalize=True).sort_index())
    
    # Prepare features
    feature_groups = config['features'].get('groups')
    df_features, group_map = prepare_features(df, groups=feature_groups, horizon=horizon)
    
    # Split by time
    train_df, val_df, test_df = split_data_by_time(
        df_features,
        train_end=config['data']['train_end'],
        test_start=config['data']['test_start'],
        val_ratio=0.15
    )
    
    # Get feature columns
    feature_cols = get_indicator_columns(df_features,
                                         ['time', 'target', 'smoothed_close', 'smooth_slope'])
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    print(f"\n📊 Features: {len(feature_cols)}")
    
    # Get X, y
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values.astype(int)
    
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values.astype(int)
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values.astype(int)
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # Create and train model
    arch_config = config['architecture']
    train_config = config['training']
    
    model = CNNLSTMModel(
        n_classes=config['labeling'].get('n_classes', 3),
        lookback=config['features'].get('lookback', 20),
        conv_filters=arch_config['conv_filters'],
        lstm_units=arch_config['lstm_units'],
        dropout=arch_config['dropout'],
        dense_units=arch_config['dense_units'],
        learning_rate=train_config['learning_rate'],
        device=train_config.get('device', 'cuda'),
        random_seed=train_config.get('random_seed', 42)
    )
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_cols,
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        patience=train_config['patience']
    )
    
    # Evaluate on test
    print("\n" + "="*60)
    print("📊 Test Set Evaluation")
    print("="*60)
    
    metrics = model.evaluate(X_test, y_test)
    
    print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"F1 Macro:    {metrics['f1_macro']:.4f}")
    
    # Save model
    output_dir = Path(config['output']['model_dir'])
    model.save(output_dir, name=f'cnn_lstm_h{horizon}')
    
    # Add horizon to metrics
    metrics['horizon'] = horizon
    metrics['model_type'] = 'cnn_lstm'
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Bitcoin price prediction models')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--horizon', type=int, default=None, help='Override prediction horizon')
    parser.add_argument('--dry-run', action='store_true', help='Validate config only')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override horizon if specified
    horizon = args.horizon if args.horizon is not None else config['features'].get('horizon', 1)
    
    print("\n" + "="*60)
    print(f"📋 Config: {args.config}")
    print(f"📊 Model type: {config['model']['type']}")
    print(f"📈 Horizon: {horizon} bars")
    print("="*60)
    
    if args.dry_run:
        print("\n✅ Config validated. Dry run complete.")
        return
    
    # Train based on model type
    model_type = config['model']['type']
    
    if model_type == 'xgboost':
        metrics = train_baseline(config, horizon=horizon)
    elif model_type == 'cnn_lstm':
        metrics = train_cnn_lstm(config, horizon=horizon)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save metrics
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / f'{model_type}_h{horizon}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to {metrics_file}")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
