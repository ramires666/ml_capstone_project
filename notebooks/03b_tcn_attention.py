"""
==============================================================================
03b - TCN-ATTENTION HYPERPARAMETER TUNING (Grid Search)
==============================================================================

This script runs a randomized grid search over key TCN-Attention hyperparameters
to find the best configuration for our Bitcoin price direction task.

Search axes (based on expert recommendations):
- tcn_filters: 32, 64, 128 (model capacity)
- num_tcn_blocks: 2, 3, 4 (network depth)  
- lookback: 16, 32, 64 (how much history to consider)
- dropout: 0.1, 0.2, 0.3 (regularization strength)
- use_class_weights: True, False (handle class imbalance)

We sample 20 random combinations from 162 total to keep runtime reasonable.
Each config takes ~2-3 minutes, so full search is ~40-60 min.
"""

# %%
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import itertools
import random
import json
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.tcn_attention import TCNAttentionModel

print("✅ Imports loaded")

# %%
# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Oracle labeling params - must match other notebooks for fair comparison
SIGMA = 3
THRESHOLD = 0.0002
HORIZON = 1
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
MODEL_DIR = 'models_artifacts'

# Hyperparameter search space
PARAM_GRID = {
    'tcn_filters': [32, 64, 128],
    'num_tcn_blocks': [2, 3, 4],
    'lookback': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'use_class_weights': [True, False]
}

# How many random combinations to try (full grid = 162 combos)
N_RANDOM_SAMPLES = 20

# Fixed params that we don't search over
FIXED_PARAMS = {
    'kernel_size': 3,
    'attention_heads': 4,
    'dense_units': 32,
    'learning_rate': 0.0007,
    'epochs': 30,
    'patience': 5,
    'batch_size': 128
}

print("📋 Grid Search Configuration:")
print(f"   Random samples: {N_RANDOM_SAMPLES}")
print(f"   Parameter grid: {list(PARAM_GRID.keys())}")

# %%
# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n" + "="*60)
print("📥 LOADING DATA")
print("="*60)

print("   Loading raw OHLCV data...")
df = load_and_merge_data(end_date='2025-12-31')
print(f"   ✓ Loaded {len(df):,} candles")

print("   Creating oracle labels...")
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)

print("   Building feature matrix...")
df_features, _ = prepare_features(df, horizon=HORIZON)

print("   Splitting into train/val/test...")
train_df, val_df, test_df = split_data_by_time(
    df_features, train_end=TRAIN_END, test_start=TEST_START, val_ratio=0.1
)

# Get feature column names (exclude target and metadata)
feature_cols = get_indicator_columns(
    df_features, exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
)
feature_cols = [c for c in feature_cols if c in train_df.columns]

# Convert to numpy, replacing any NaN/inf with 0
X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_train = train_df['target'].values.astype(int)
X_val = np.nan_to_num(val_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_val = val_df['target'].values.astype(int)
X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_test = test_df['target'].values.astype(int)

print(f"\n   Train: {X_train.shape[0]:,} samples")
print(f"   Val:   {X_val.shape[0]:,} samples")
print(f"   Test:  {X_test.shape[0]:,} samples")
print(f"   Features: {len(feature_cols)}")

# %%
# ==============================================================================
# GENERATE RANDOM COMBINATIONS
# ==============================================================================

# Generate all possible parameter combinations
all_combinations = list(itertools.product(*PARAM_GRID.values()))
param_names = list(PARAM_GRID.keys())

print(f"\n📊 Total possible combinations: {len(all_combinations)}")
print(f"   Sampling {N_RANDOM_SAMPLES} random combinations...")

# Random sample for efficiency (full grid would take too long)
random.seed(42)
sampled_indices = random.sample(range(len(all_combinations)), min(N_RANDOM_SAMPLES, len(all_combinations)))
sampled_combinations = [all_combinations[i] for i in sampled_indices]

# Convert tuples to dicts for easier handling
param_configs = []
for combo in sampled_combinations:
    config = dict(zip(param_names, combo))
    param_configs.append(config)

print(f"   ✓ Selected {len(param_configs)} configs to test")

# %%
# ==============================================================================
# RUN GRID SEARCH
# ==============================================================================

print("\n" + "="*60)
print("🔍 RUNNING GRID SEARCH")
print("="*60)

results = []

for i, params in enumerate(param_configs):
    print(f"\n{'─'*60}")
    print(f"[{i+1}/{len(param_configs)}] Testing config:")
    print(f"   filters={params['tcn_filters']}, blocks={params['num_tcn_blocks']}, "
          f"lookback={params['lookback']}, dropout={params['dropout']}, "
          f"class_weights={params['use_class_weights']}")
    print("─"*60)
    
    try:
        # Create model with current hyperparameters
        print("   Creating model...")
        model = TCNAttentionModel(
            n_classes=3,
            lookback=params['lookback'],
            tcn_filters=params['tcn_filters'],
            num_tcn_blocks=params['num_tcn_blocks'],
            dropout=params['dropout'],
            kernel_size=FIXED_PARAMS['kernel_size'],
            attention_heads=FIXED_PARAMS['attention_heads'],
            dense_units=FIXED_PARAMS['dense_units'],
            learning_rate=FIXED_PARAMS['learning_rate'],
            device='cuda'
        )
        
        # Train the model
        print("   Training...")
        model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=FIXED_PARAMS['epochs'],
            batch_size=FIXED_PARAMS['batch_size'],
            patience=FIXED_PARAMS['patience'],
            use_class_weights=params['use_class_weights']
        )
        
        # Evaluate on validation set (not test - save that for final model)
        print("   Evaluating on validation set...")
        val_metrics = model.evaluate(X_val, y_val)
        
        # Store results
        result = {
            **params,
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_weighted': val_metrics['f1_weighted'],
            'val_f1_macro': val_metrics['f1_macro']
        }
        results.append(result)
        
        print(f"   ✅ val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1_weighted']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({**params, 'val_accuracy': 0, 'val_f1_weighted': 0, 'val_f1_macro': 0, 'error': str(e)})

# %%
# ==============================================================================
# ANALYZE RESULTS
# ==============================================================================

print("\n" + "="*60)
print("📊 GRID SEARCH RESULTS")
print("="*60)

# Sort by validation F1 score (our main metric)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_f1_weighted', ascending=False)

print("\n🏆 TOP 5 CONFIGURATIONS (by val_f1_weighted):\n")
top5 = results_df.head(5)
for idx, row in top5.iterrows():
    print(f"  F1={row['val_f1_weighted']:.4f} | "
          f"filters={int(row['tcn_filters'])}, blocks={int(row['num_tcn_blocks'])}, "
          f"lookback={int(row['lookback'])}, dropout={row['dropout']}, "
          f"class_weights={row['use_class_weights']}")

# Extract best params
best_params = results_df.iloc[0].to_dict()
print(f"\n🥇 BEST CONFIGURATION:")
print(f"   tcn_filters:      {int(best_params['tcn_filters'])}")
print(f"   num_tcn_blocks:   {int(best_params['num_tcn_blocks'])}")
print(f"   lookback:         {int(best_params['lookback'])}")
print(f"   dropout:          {best_params['dropout']}")
print(f"   use_class_weights: {best_params['use_class_weights']}")
print(f"   val_f1_weighted:  {best_params['val_f1_weighted']:.4f}")

# %%
# ==============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMS
# ==============================================================================

print("\n" + "="*60)
print("🏆 TRAINING FINAL MODEL WITH BEST PARAMS")
print("="*60)

print("   Creating model with best hyperparameters...")
best_model = TCNAttentionModel(
    n_classes=3,
    lookback=int(best_params['lookback']),
    tcn_filters=int(best_params['tcn_filters']),
    num_tcn_blocks=int(best_params['num_tcn_blocks']),
    dropout=best_params['dropout'],
    kernel_size=FIXED_PARAMS['kernel_size'],
    attention_heads=FIXED_PARAMS['attention_heads'],
    dense_units=FIXED_PARAMS['dense_units'],
    learning_rate=FIXED_PARAMS['learning_rate'],
    device='cuda'
)

print("   Training final model...")
best_model.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=FIXED_PARAMS['epochs'],
    batch_size=FIXED_PARAMS['batch_size'],
    patience=FIXED_PARAMS['patience'],
    use_class_weights=best_params['use_class_weights']
)

# Now we can evaluate on the held-out test set
print("   Evaluating on test set (held out until now)...")
test_metrics = best_model.evaluate(X_test, y_test)

print(f"\n✅ FINAL TEST SET RESULTS:")
print(f"   Accuracy:    {test_metrics['accuracy']:.4f}")
print(f"   F1 Weighted: {test_metrics['f1_weighted']:.4f}")
print(f"   F1 Macro:    {test_metrics['f1_macro']:.4f}")

# %%
# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "="*60)
print("💾 SAVING RESULTS")
print("="*60)

# Save the trained model
print("   Saving model weights...")
Path(MODEL_DIR).mkdir(exist_ok=True)
best_model.save(MODEL_DIR, name=f'tcn_attention_h{HORIZON}')

# Save grid search results as CSV for later analysis
print("   Saving grid search results...")
results_df.to_csv(f'{MODEL_DIR}/tcn_attention_grid_search_results.csv', index=False)

# Save best params as JSON for notebook 04 to load
print("   Saving best parameters...")
best_params_clean = {
    'tcn_filters': int(best_params['tcn_filters']),
    'num_tcn_blocks': int(best_params['num_tcn_blocks']),
    'lookback': int(best_params['lookback']),
    'dropout': float(best_params['dropout']),
    'use_class_weights': bool(best_params['use_class_weights']),
    'val_f1_weighted': float(best_params['val_f1_weighted']),
    'test_accuracy': float(test_metrics['accuracy']),
    'test_f1_weighted': float(test_metrics['f1_weighted'])
}

with open(f'{MODEL_DIR}/tcn_attention_best_params.json', 'w') as f:
    json.dump(best_params_clean, f, indent=2)

print(f"\n   ✅ Model: {MODEL_DIR}/tcn_attention_h{HORIZON}_model.keras")
print(f"   ✅ Grid results: {MODEL_DIR}/tcn_attention_grid_search_results.csv")
print(f"   ✅ Best params: {MODEL_DIR}/tcn_attention_best_params.json")

# %%
# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)

print(f"""
Grid Search Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Tested {len(results)} configurations
• Best val F1 weighted: {best_params['val_f1_weighted']:.4f}

Best Hyperparameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• tcn_filters:      {int(best_params['tcn_filters'])}
• num_tcn_blocks:   {int(best_params['num_tcn_blocks'])}
• lookback:         {int(best_params['lookback'])}
• dropout:          {best_params['dropout']}
• use_class_weights: {best_params['use_class_weights']}

Final Test Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Accuracy:    {test_metrics['accuracy']:.4f}
• F1 Weighted: {test_metrics['f1_weighted']:.4f}
• F1 Macro:    {test_metrics['f1_macro']:.4f}

Next: Run 05_comparison to compare with XGBoost and CNN-LSTM.
""")
