"""
==============================================================================
04 - TRAIN MODELS FOR ADDITIONAL HORIZONS (3 and 5 bars ahead)
==============================================================================

PURPOSE OF THIS NOTEBOOK:
-------------------------
Train XGBoost and CNN-LSTM models for horizons 3 and 5.
Horizon 1 models are already trained in notebooks 02 and 03.

RUN ORDER:
----------
Each cell can be run independently after the data loading cells.
This allows you to control long-running training step by step.

ESTIMATED TIME:
---------------
- Data loading: ~1 minute
- Each XGBoost: ~5-10 minutes  
- Each CNN-LSTM: ~5-15 minutes
- Total: ~30-60 minutes for all 4 models
"""

# %% [markdown]
# # 04 - Train Additional Horizons
# 
# This notebook trains models for horizons 3 and 5.
# Each training step is a separate cell for easy debugging.

# %%
# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

import sys
from pathlib import Path

project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.xgb import XGBBaseline
from src.models.cnn_lstm import CNNLSTMModel

print("✅ Imports loaded")

# %%
# ==============================================================================
# CONFIGURATION
# ==============================================================================

SIGMA = 4
THRESHOLD = 0.0002
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
MODEL_DIR = 'models_artifacts'

# Store all results for final summary
all_results = {}

print("📋 Configuration:")
print(f"   Oracle: sigma={SIGMA}, threshold={THRESHOLD}")
print(f"   Train period: up to {TRAIN_END}")
print(f"   Test period: from {TEST_START}")

# %%
# ==============================================================================
# LOAD DATA (run once, used by all training cells)
# ==============================================================================

print("=" * 60)
print("📥 LOADING DATA")
print("=" * 60)

df = load_and_merge_data(end_date='2025-12-31')
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)

print(f"✅ Loaded {len(df):,} rows")

# Show label distribution
label_dist = df['target'].value_counts(normalize=True).sort_index()
label_names = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
print("\nLabel distribution:")
for label, pct in label_dist.items():
    print(f"   {label_names[label]:8s}: {pct*100:5.1f}%")

# %% [markdown]
# ---
# ## HORIZON 3 (45 minutes ahead)
# ---

# %%
# ==============================================================================
# PREPARE DATA FOR HORIZON 3
# ==============================================================================

HORIZON = 3
print(f"\n{'='*60}")
print(f"🎯 PREPARING DATA FOR HORIZON={HORIZON} ({HORIZON * 15} minutes ahead)")
print("="*60)

all_results[HORIZON] = {}

# Prepare features with horizon shift
df_features_h3, group_map = prepare_features(df.copy(), horizon=HORIZON)
print(f"   Generated {sum(len(c) for c in group_map.values())} features")

# Split data
train_df_h3, val_df_h3, test_df_h3 = split_data_by_time(
    df_features_h3, train_end=TRAIN_END, test_start=TEST_START, val_ratio=0.1
)
print(f"   Train: {len(train_df_h3):,}, Val: {len(val_df_h3):,}, Test: {len(test_df_h3):,}")

# Get feature columns
feature_cols = get_indicator_columns(
    df_features_h3, exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
)
feature_cols = [c for c in feature_cols if c in train_df_h3.columns]

# Convert to numpy
X_train_h3 = np.nan_to_num(train_df_h3[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_train_h3 = train_df_h3['target'].values.astype(int)
X_val_h3 = np.nan_to_num(val_df_h3[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_val_h3 = val_df_h3['target'].values.astype(int)
X_test_h3 = np.nan_to_num(test_df_h3[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_test_h3 = test_df_h3['target'].values.astype(int)

print("✅ Data ready for horizon 3")

# %%
# ==============================================================================
# TRAIN XGBOOST FOR HORIZON 3
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🌲 TRAINING XGBOOST (Horizon={HORIZON})")
print("─"*60)

xgb_h3 = XGBBaseline(n_classes=3, device='cuda', random_state=42)

print("[1/3] Training...")
xgb_h3.fit(X_train_h3, y_train_h3, X_val_h3, y_val_h3, feature_names=feature_cols)

print("[2/3] Tuning hyperparameters...")
best_params = xgb_h3.tune(X_train_h3, y_train_h3, n_iter=15, cv_splits=3, scoring='f1_weighted')

print("\n[3/3] Evaluating...")
xgb_metrics_h3 = xgb_h3.evaluate(X_test_h3, y_test_h3)

print(f"\n✅ XGBoost H=3 Results:")
print(f"   Accuracy: {xgb_metrics_h3['accuracy']:.4f}")
print(f"   F1 Weighted: {xgb_metrics_h3['f1_weighted']:.4f}")

# Save
Path(MODEL_DIR).mkdir(exist_ok=True)
xgb_h3.save(MODEL_DIR, name='xgb_baseline_h3')
all_results[3]['XGBoost'] = xgb_metrics_h3
print(f"💾 Saved: {MODEL_DIR}/xgb_baseline_h3_model.joblib")

# %%
# ==============================================================================
# TRAIN CNN-LSTM FOR HORIZON 3
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🧠 TRAINING CNN-LSTM (Horizon={HORIZON})")
print("─"*60)

cnn_h3 = CNNLSTMModel(
    n_classes=3,
    lookback=32,
    conv_filters=64,
    lstm_units=64,
    dropout=0.3,
    learning_rate=0.0007,  # Reduced 30% for stability
    device='cuda'
)

print("Training with early stopping (patience=5)...")
history = cnn_h3.fit(
    X_train_h3, y_train_h3,
    X_val_h3, y_val_h3,
    epochs=30,
    batch_size=128,
    patience=5
)

cnn_metrics_h3 = cnn_h3.evaluate(X_test_h3, y_test_h3)

print(f"\n✅ CNN-LSTM H=3 Results:")
print(f"   Accuracy: {cnn_metrics_h3['accuracy']:.4f}")
print(f"   F1 Weighted: {cnn_metrics_h3['f1_weighted']:.4f}")

cnn_h3.save(MODEL_DIR, name='cnn_lstm_h3')
all_results[3]['CNN-LSTM'] = cnn_metrics_h3
print(f"💾 Saved: {MODEL_DIR}/cnn_lstm_h3_model.keras")

# %% [markdown]
# ---
# ## HORIZON 5 (75 minutes ahead)
# ---

# %%
# ==============================================================================
# PREPARE DATA FOR HORIZON 5
# ==============================================================================

HORIZON = 5
print(f"\n{'='*60}")
print(f"🎯 PREPARING DATA FOR HORIZON={HORIZON} ({HORIZON * 15} minutes ahead)")
print("="*60)

all_results[HORIZON] = {}

# Prepare features with horizon shift
df_features_h5, group_map = prepare_features(df.copy(), horizon=HORIZON)
print(f"   Generated {sum(len(c) for c in group_map.values())} features")

# Split data
train_df_h5, val_df_h5, test_df_h5 = split_data_by_time(
    df_features_h5, train_end=TRAIN_END, test_start=TEST_START, val_ratio=0.1
)
print(f"   Train: {len(train_df_h5):,}, Val: {len(val_df_h5):,}, Test: {len(test_df_h5):,}")

# Get feature columns
feature_cols = get_indicator_columns(
    df_features_h5, exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
)
feature_cols = [c for c in feature_cols if c in train_df_h5.columns]

# Convert to numpy
X_train_h5 = np.nan_to_num(train_df_h5[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_train_h5 = train_df_h5['target'].values.astype(int)
X_val_h5 = np.nan_to_num(val_df_h5[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_val_h5 = val_df_h5['target'].values.astype(int)
X_test_h5 = np.nan_to_num(test_df_h5[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_test_h5 = test_df_h5['target'].values.astype(int)

print("✅ Data ready for horizon 5")

# %%
# ==============================================================================
# TRAIN XGBOOST FOR HORIZON 5
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🌲 TRAINING XGBOOST (Horizon={HORIZON})")
print("─"*60)

xgb_h5 = XGBBaseline(n_classes=3, device='cuda', random_state=42)

print("[1/3] Training...")
xgb_h5.fit(X_train_h5, y_train_h5, X_val_h5, y_val_h5, feature_names=feature_cols)

print("[2/3] Tuning hyperparameters...")
best_params = xgb_h5.tune(X_train_h5, y_train_h5, n_iter=15, cv_splits=3, scoring='f1_weighted')

print("\n[3/3] Evaluating...")
xgb_metrics_h5 = xgb_h5.evaluate(X_test_h5, y_test_h5)

print(f"\n✅ XGBoost H=5 Results:")
print(f"   Accuracy: {xgb_metrics_h5['accuracy']:.4f}")
print(f"   F1 Weighted: {xgb_metrics_h5['f1_weighted']:.4f}")

xgb_h5.save(MODEL_DIR, name='xgb_baseline_h5')
all_results[5]['XGBoost'] = xgb_metrics_h5
print(f"💾 Saved: {MODEL_DIR}/xgb_baseline_h5_model.joblib")

# %%
# ==============================================================================
# TRAIN CNN-LSTM FOR HORIZON 5
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🧠 TRAINING CNN-LSTM (Horizon={HORIZON})")
print("─"*60)

cnn_h5 = CNNLSTMModel(
    n_classes=3,
    lookback=32,
    conv_filters=64,
    lstm_units=64,
    dropout=0.3,
    learning_rate=0.0007,
    device='cuda'
)

print("Training with early stopping (patience=5)...")
history = cnn_h5.fit(
    X_train_h5, y_train_h5,
    X_val_h5, y_val_h5,
    epochs=30,
    batch_size=128,
    patience=5
)

cnn_metrics_h5 = cnn_h5.evaluate(X_test_h5, y_test_h5)

print(f"\n✅ CNN-LSTM H=5 Results:")
print(f"   Accuracy: {cnn_metrics_h5['accuracy']:.4f}")
print(f"   F1 Weighted: {cnn_metrics_h5['f1_weighted']:.4f}")

cnn_h5.save(MODEL_DIR, name='cnn_lstm_h5')
all_results[5]['CNN-LSTM'] = cnn_metrics_h5
print(f"💾 Saved: {MODEL_DIR}/cnn_lstm_h5_model.keras")

# %% [markdown]
# ---
# ## FINAL SUMMARY
# ---

# %%
# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("📋 TRAINING COMPLETE - SUMMARY")
print("=" * 60)

print(f"\n{'Horizon':<10} {'Model':<12} {'Accuracy':<12} {'F1 Weighted':<12}")
print("─" * 48)

for h in [3, 5]:
    if h in all_results:
        for model in ['XGBoost', 'CNN-LSTM']:
            if model in all_results[h]:
                m = all_results[h][model]
                print(f"{h:<10} {model:<12} {m['accuracy']:.4f}       {m['f1_weighted']:.4f}")

print(f"""
{'─' * 48}

📁 Models saved to: {MODEL_DIR}/
   - xgb_baseline_h3_model.joblib
   - xgb_baseline_h5_model.joblib
   - cnn_lstm_h3_model.keras
   - cnn_lstm_h5_model.keras

🔜 NEXT STEP: Run notebook 05_comparison to see full comparison.

✅ ALL DONE!
""")
