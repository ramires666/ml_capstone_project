"""
==============================================================================
04 - TRAIN MODELS FOR ADDITIONAL HORIZONS (3 and 5 bars ahead)
==============================================================================

Train XGBoost, CNN-LSTM, and TCN-Attention for horizons 3 and 5.
Horizon 1 models trained in notebooks 02, 03, 03b.

Each training cell is independent after data loading.
"""

# %% [markdown]
# # 04 - Train Additional Horizons
# 
# Train all 3 model types for horizons 3 and 5.

# %%
# ==============================================================================
# IMPORTS
# ==============================================================================

import sys
from pathlib import Path

project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.xgb import XGBBaseline
from src.models.cnn_lstm import CNNLSTMModel
from src.models.tcn_attention import TCNAttentionModel

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

# Load best TCN params if available (from grid search)
try:
    with open(f'{MODEL_DIR}/tcn_attention_best_params.json', 'r') as f:
        TCN_BEST_PARAMS = json.load(f)
    print(f"📋 Loaded TCN best params from grid search")
except:
    # Default params if grid search not run
    TCN_BEST_PARAMS = {
        'tcn_filters': 64,
        'num_tcn_blocks': 3,
        'lookback': 32,
        'dropout': 0.2,
        'use_class_weights': False
    }
    print(f"📋 Using default TCN params (run 03b for tuned params)")

all_results = {}

print(f"   Oracle: sigma={SIGMA}, threshold={THRESHOLD}")

# %%
# ==============================================================================
# LOAD DATA
# ==============================================================================

print("=" * 60)
print("📥 LOADING DATA")
print("=" * 60)

df = load_and_merge_data(end_date='2025-12-31')
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)
print(f"✅ Loaded {len(df):,} rows")

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
print(f"🎯 PREPARING DATA FOR HORIZON={HORIZON}")
print("="*60)

all_results[HORIZON] = {}

df_features_h3, _ = prepare_features(df.copy(), horizon=HORIZON)
train_df_h3, val_df_h3, test_df_h3 = split_data_by_time(
    df_features_h3, train_end=TRAIN_END, test_start=TEST_START, val_ratio=0.1
)

feature_cols = get_indicator_columns(
    df_features_h3, exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
)
feature_cols = [c for c in feature_cols if c in train_df_h3.columns]

X_train_h3 = np.nan_to_num(train_df_h3[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_train_h3 = train_df_h3['target'].values.astype(int)
X_val_h3 = np.nan_to_num(val_df_h3[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_val_h3 = val_df_h3['target'].values.astype(int)
X_test_h3 = np.nan_to_num(test_df_h3[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_test_h3 = test_df_h3['target'].values.astype(int)

print(f"   Train: {len(train_df_h3):,}, Val: {len(val_df_h3):,}, Test: {len(test_df_h3):,}")

# %%
# ==============================================================================
# TRAIN XGBOOST FOR HORIZON 3
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🌲 TRAINING XGBOOST (H={HORIZON})")
print("─"*60)

xgb_h3 = XGBBaseline(n_classes=3, device='cuda', random_state=42)
xgb_h3.fit(X_train_h3, y_train_h3, X_val_h3, y_val_h3, feature_names=feature_cols)
xgb_h3.tune(X_train_h3, y_train_h3, n_iter=15, cv_splits=3, scoring='f1_weighted')
xgb_metrics_h3 = xgb_h3.evaluate(X_test_h3, y_test_h3)

print(f"✅ Acc={xgb_metrics_h3['accuracy']:.4f}, F1={xgb_metrics_h3['f1_weighted']:.4f}")
Path(MODEL_DIR).mkdir(exist_ok=True)
xgb_h3.save(MODEL_DIR, name='xgb_baseline_h3')
all_results[3]['XGBoost'] = xgb_metrics_h3

# %%
# ==============================================================================
# TRAIN CNN-LSTM FOR HORIZON 3
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🧠 TRAINING CNN-LSTM (H={HORIZON})")
print("─"*60)

cnn_h3 = CNNLSTMModel(
    n_classes=3, lookback=32, conv_filters=64, lstm_units=64,
    dropout=0.3, learning_rate=0.0007, device='cuda'
)
cnn_h3.fit(X_train_h3, y_train_h3, X_val_h3, y_val_h3, epochs=30, batch_size=128, patience=5)
cnn_metrics_h3 = cnn_h3.evaluate(X_test_h3, y_test_h3)

print(f"✅ Acc={cnn_metrics_h3['accuracy']:.4f}, F1={cnn_metrics_h3['f1_weighted']:.4f}")
cnn_h3.save(MODEL_DIR, name='cnn_lstm_h3')
all_results[3]['CNN-LSTM'] = cnn_metrics_h3

# %%
# ==============================================================================
# TRAIN TCN-ATTENTION FOR HORIZON 3
# ==============================================================================

print(f"\n{'─'*60}")
print(f"⚡ TRAINING TCN-ATTENTION (H={HORIZON})")
print("─"*60)

tcn_h3 = TCNAttentionModel(
    n_classes=3,
    lookback=TCN_BEST_PARAMS.get('lookback', 32),
    tcn_filters=TCN_BEST_PARAMS.get('tcn_filters', 64),
    num_tcn_blocks=TCN_BEST_PARAMS.get('num_tcn_blocks', 3),
    dropout=TCN_BEST_PARAMS.get('dropout', 0.2),
    learning_rate=0.0007,
    device='cuda'
)
tcn_h3.fit(
    X_train_h3, y_train_h3, X_val_h3, y_val_h3,
    epochs=30, batch_size=128, patience=5,
    use_class_weights=TCN_BEST_PARAMS.get('use_class_weights', False)
)
tcn_metrics_h3 = tcn_h3.evaluate(X_test_h3, y_test_h3)

print(f"✅ Acc={tcn_metrics_h3['accuracy']:.4f}, F1={tcn_metrics_h3['f1_weighted']:.4f}")
tcn_h3.save(MODEL_DIR, name='tcn_attention_h3')
all_results[3]['TCN-Attention'] = tcn_metrics_h3

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
print(f"🎯 PREPARING DATA FOR HORIZON={HORIZON}")
print("="*60)

all_results[HORIZON] = {}

df_features_h5, _ = prepare_features(df.copy(), horizon=HORIZON)
train_df_h5, val_df_h5, test_df_h5 = split_data_by_time(
    df_features_h5, train_end=TRAIN_END, test_start=TEST_START, val_ratio=0.1
)

feature_cols = get_indicator_columns(
    df_features_h5, exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
)
feature_cols = [c for c in feature_cols if c in train_df_h5.columns]

X_train_h5 = np.nan_to_num(train_df_h5[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_train_h5 = train_df_h5['target'].values.astype(int)
X_val_h5 = np.nan_to_num(val_df_h5[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_val_h5 = val_df_h5['target'].values.astype(int)
X_test_h5 = np.nan_to_num(test_df_h5[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_test_h5 = test_df_h5['target'].values.astype(int)

print(f"   Train: {len(train_df_h5):,}, Val: {len(val_df_h5):,}, Test: {len(test_df_h5):,}")

# %%
# ==============================================================================
# TRAIN XGBOOST FOR HORIZON 5
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🌲 TRAINING XGBOOST (H={HORIZON})")
print("─"*60)

xgb_h5 = XGBBaseline(n_classes=3, device='cuda', random_state=42)
xgb_h5.fit(X_train_h5, y_train_h5, X_val_h5, y_val_h5, feature_names=feature_cols)
xgb_h5.tune(X_train_h5, y_train_h5, n_iter=15, cv_splits=3, scoring='f1_weighted')
xgb_metrics_h5 = xgb_h5.evaluate(X_test_h5, y_test_h5)

print(f"✅ Acc={xgb_metrics_h5['accuracy']:.4f}, F1={xgb_metrics_h5['f1_weighted']:.4f}")
xgb_h5.save(MODEL_DIR, name='xgb_baseline_h5')
all_results[5]['XGBoost'] = xgb_metrics_h5

# %%
# ==============================================================================
# TRAIN CNN-LSTM FOR HORIZON 5
# ==============================================================================

print(f"\n{'─'*60}")
print(f"🧠 TRAINING CNN-LSTM (H={HORIZON})")
print("─"*60)

cnn_h5 = CNNLSTMModel(
    n_classes=3, lookback=32, conv_filters=64, lstm_units=64,
    dropout=0.3, learning_rate=0.0007, device='cuda'
)
cnn_h5.fit(X_train_h5, y_train_h5, X_val_h5, y_val_h5, epochs=30, batch_size=128, patience=5)
cnn_metrics_h5 = cnn_h5.evaluate(X_test_h5, y_test_h5)

print(f"✅ Acc={cnn_metrics_h5['accuracy']:.4f}, F1={cnn_metrics_h5['f1_weighted']:.4f}")
cnn_h5.save(MODEL_DIR, name='cnn_lstm_h5')
all_results[5]['CNN-LSTM'] = cnn_metrics_h5

# %%
# ==============================================================================
# TRAIN TCN-ATTENTION FOR HORIZON 5
# ==============================================================================

print(f"\n{'─'*60}")
print(f"⚡ TRAINING TCN-ATTENTION (H={HORIZON})")
print("─"*60)

tcn_h5 = TCNAttentionModel(
    n_classes=3,
    lookback=TCN_BEST_PARAMS.get('lookback', 32),
    tcn_filters=TCN_BEST_PARAMS.get('tcn_filters', 64),
    num_tcn_blocks=TCN_BEST_PARAMS.get('num_tcn_blocks', 3),
    dropout=TCN_BEST_PARAMS.get('dropout', 0.2),
    learning_rate=0.0007,
    device='cuda'
)
tcn_h5.fit(
    X_train_h5, y_train_h5, X_val_h5, y_val_h5,
    epochs=30, batch_size=128, patience=5,
    use_class_weights=TCN_BEST_PARAMS.get('use_class_weights', False)
)
tcn_metrics_h5 = tcn_h5.evaluate(X_test_h5, y_test_h5)

print(f"✅ Acc={tcn_metrics_h5['accuracy']:.4f}, F1={tcn_metrics_h5['f1_weighted']:.4f}")
tcn_h5.save(MODEL_DIR, name='tcn_attention_h5')
all_results[5]['TCN-Attention'] = tcn_metrics_h5

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

print(f"\n{'Horizon':<10} {'Model':<15} {'Accuracy':<12} {'F1 Weighted':<12}")
print("─" * 50)

for h in [3, 5]:
    if h in all_results:
        for model in ['XGBoost', 'CNN-LSTM', 'TCN-Attention']:
            if model in all_results[h]:
                m = all_results[h][model]
                print(f"{h:<10} {model:<15} {m['accuracy']:.4f}       {m['f1_weighted']:.4f}")

print(f"""
{'─' * 50}

📁 Models saved to: {MODEL_DIR}/
   • xgb_baseline_h3/h5
   • cnn_lstm_h3/h5
   • tcn_attention_h3/h5

🔜 NEXT: Run 05_comparison for full comparison.
""")
