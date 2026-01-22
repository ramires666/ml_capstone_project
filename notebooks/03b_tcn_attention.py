"""
==============================================================================
03b - TCN-ATTENTION MODEL (Alternative to CNN-LSTM)
==============================================================================

This notebook trains a TCN-Attention model as an alternative architecture.
Run after 03_cnn_lstm to compare both approaches.

Key differences from CNN-LSTM:
- TCN: dilated causal convolutions instead of LSTM
- Attention: learns which time steps matter most
- Class weights: handles imbalanced UP/DOWN/SIDEWAYS
- Optimized for F1 score, not just accuracy
"""

# %% [markdown]
# # 03b - TCN-Attention Model
# 
# Alternative to CNN-LSTM using Temporal Convolutional Networks + Attention.

# %%
# ==============================================================================
# IMPORTS
# ==============================================================================

import sys
from pathlib import Path

project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
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

# Same parameters as other notebooks for fair comparison
SIGMA = 4
THRESHOLD = 0.0002
HORIZON = 1
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
MODEL_DIR = 'models_artifacts'

print("📋 Configuration:")
print(f"   Horizon: {HORIZON}")
print(f"   Oracle: sigma={SIGMA}, threshold={THRESHOLD}")

# %%
# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

print("\n" + "="*60)
print("📥 LOADING DATA")
print("="*60)

df = load_and_merge_data(end_date='2025-12-31')
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)
df_features, _ = prepare_features(df, horizon=HORIZON)

train_df, val_df, test_df = split_data_by_time(
    df_features, train_end=TRAIN_END, test_start=TEST_START, val_ratio=0.1
)

feature_cols = get_indicator_columns(
    df_features, exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
)
feature_cols = [c for c in feature_cols if c in train_df.columns]

X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_train = train_df['target'].values.astype(int)
X_val = np.nan_to_num(val_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_val = val_df['target'].values.astype(int)
X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
y_test = test_df['target'].values.astype(int)

print(f"   Train: {X_train.shape[0]:,} samples")
print(f"   Val: {X_val.shape[0]:,} samples")
print(f"   Test: {X_test.shape[0]:,} samples")

# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
print("\n   Class distribution in train:")
for u, c in zip(unique, counts):
    label = ['DOWN', 'SIDEWAYS', 'UP'][u]
    print(f"      {label}: {c:,} ({c/len(y_train)*100:.1f}%)")

# %%
# ==============================================================================
# TRAIN TCN-ATTENTION MODEL
# ==============================================================================

print("\n" + "="*60)
print("🧠 TRAINING TCN-ATTENTION MODEL")
print("="*60)

# Model hyperparameters
# TCN with attention tends to need less aggressive regularization
model = TCNAttentionModel(
    n_classes=3,
    lookback=32,           # Same as CNN-LSTM for fair comparison
    tcn_filters=64,        # Number of filters in TCN blocks
    kernel_size=3,         # Convolution kernel size
    num_tcn_blocks=3,      # Stacked TCN blocks (dilation: 1, 2, 4)
    attention_heads=4,     # Multi-head attention
    dropout=0.2,           # Lower dropout than CNN-LSTM
    dense_units=32,        # Dense layer before output
    learning_rate=0.0007,  # Same LR as CNN-LSTM
    device='cuda'
)

# Train with class weights to handle imbalance
# This helps with F1 score by not ignoring minority classes
history = model.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=30,
    batch_size=128,
    patience=5,
    use_class_weights=True  # Important for F1 optimization
)

# %%
# ==============================================================================
# EVALUATE ON TEST SET
# ==============================================================================

print("\n" + "="*60)
print("📊 TEST SET EVALUATION")
print("="*60)

metrics = model.evaluate(X_test, y_test)

print(f"\n✅ TCN-Attention Results:")
print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
print(f"   F1 Weighted: {metrics['f1_weighted']:.4f}")
print(f"   F1 Macro:    {metrics['f1_macro']:.4f}")

# Per-class breakdown
print("\n   Per-class F1:")
report = metrics['classification_report']
for cls in ['DOWN', 'SIDEWAYS', 'UP']:
    cls_data = report.get(cls, report.get(cls.lower(), {}))
    f1 = cls_data.get('f1-score', 0)
    print(f"      {cls}: {f1:.4f}")

# %%
# ==============================================================================
# COMPARE WITH CNN-LSTM
# ==============================================================================

print("\n" + "="*60)
print("📊 COMPARISON WITH CNN-LSTM")
print("="*60)

# Try to load CNN-LSTM for comparison
try:
    from src.models.cnn_lstm import CNNLSTMModel
    cnn_model = CNNLSTMModel.load(MODEL_DIR, name=f'cnn_lstm_h{HORIZON}', device='cuda')
    cnn_metrics = cnn_model.evaluate(X_test, y_test)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Weighted':<12} {'F1 Macro':<12}")
    print("─" * 56)
    print(f"{'CNN-LSTM':<20} {cnn_metrics['accuracy']:.4f}       {cnn_metrics['f1_weighted']:.4f}        {cnn_metrics['f1_macro']:.4f}")
    print(f"{'TCN-Attention':<20} {metrics['accuracy']:.4f}       {metrics['f1_weighted']:.4f}        {metrics['f1_macro']:.4f}")
    
    # Who wins?
    if metrics['f1_weighted'] > cnn_metrics['f1_weighted']:
        diff = metrics['f1_weighted'] - cnn_metrics['f1_weighted']
        print(f"\n🏆 TCN-Attention wins by {diff:.4f} F1 weighted!")
    else:
        diff = cnn_metrics['f1_weighted'] - metrics['f1_weighted']
        print(f"\n🏆 CNN-LSTM wins by {diff:.4f} F1 weighted")
        
except Exception as e:
    print(f"⚠️ Could not load CNN-LSTM for comparison: {e}")
    print("   Run 03_cnn_lstm first to enable comparison.")

# %%
# ==============================================================================
# TRAINING HISTORY PLOT
# ==============================================================================

if model.history:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(model.history['train_loss'], label='Train')
    if model.history['val_loss']:
        axes[0].plot(model.history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(model.history['train_acc'], label='Train')
    if model.history['val_acc']:
        axes[1].plot(model.history['val_acc'], label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('tcn_attention_training_history.png', dpi=150)
    plt.show()

# %%
# ==============================================================================
# SAVE MODEL
# ==============================================================================

print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

Path(MODEL_DIR).mkdir(exist_ok=True)
model.save(MODEL_DIR, name=f'tcn_attention_h{HORIZON}')

print(f"\n✅ Model saved: {MODEL_DIR}/tcn_attention_h{HORIZON}_model.keras")

# %%
# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)

print(f"""
Model: TCN-Attention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Horizon: {HORIZON} bar(s)
• Lookback: {model.lookback} steps
• TCN Filters: {model.tcn_filters}
• Attention Heads: {model.attention_heads}

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Accuracy:    {metrics['accuracy']:.4f}
• F1 Weighted: {metrics['f1_weighted']:.4f}
• F1 Macro:    {metrics['f1_macro']:.4f}

Next: Run 05_comparison to see full model comparison.
""")
