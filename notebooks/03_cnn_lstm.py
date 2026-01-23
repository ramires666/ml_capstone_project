"""
==============================================================================
03 - CNN-LSTM ADVANCED MODEL
==============================================================================

PURPOSE OF THIS NOTEBOOK:
-------------------------
Train and evaluate a CNN-LSTM neural network for price direction prediction.

This is our ADVANCED model because:
1. CNN layers capture local patterns (like candlestick patterns)
2. LSTM layers capture temporal dependencies (memory of past states)
3. Combination is powerful for sequential financial data

WHY CNN + LSTM?
---------------
- CNN: Good at finding local patterns in data (like chart patterns)
- LSTM: Good at remembering long-term dependencies
- Together: Can learn complex temporal patterns in price movements

EXPECTED RESULTS:
-----------------
- Neural networks often match or slightly beat tree-based models
- Main advantage: Can learn non-linear patterns automatically
- Main disadvantage: Slower training, needs more data

"""

# %% [markdown]
# # CNN-LSTM Advanced Model
# 
# **Goal**: Train a deep learning model that can capture temporal patterns.
# 
# **Architecture**:
# 1. Input: Sequence of N past feature vectors (lookback window)
# 2. Conv1D: Extract local patterns
# 3. LSTM: Capture temporal dependencies
# 4. Dense: Classification into UP/SIDEWAYS/DOWN

# %%
# ==============================================================================
# IMPORTS AND GPU CHECK
# ==============================================================================
#
# TensorFlow/Keras is used for the neural network.
# We check for GPU availability because training is MUCH faster on GPU.

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ==============================================================================
# GPU AVAILABILITY CHECK
# ==============================================================================
# Neural networks are 10-100x faster on GPU compared to CPU.
# If no GPU is detected, training will still work but be slower.

print("="*60)
print("🖥️ SYSTEM CHECK")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {gpus[0].name}")
    print("   Training will be fast!")
else:
    print("⚠️ No GPU detected - training will be slower")
    print("   Consider using Google Colab for faster training")

# %%
# ==============================================================================
# IMPORT PROJECT MODULES
# ==============================================================================

# Reload modules to pick up any code changes without restarting kernel
import importlib
import src.data.loader
import src.labeling.oracle
import src.features.builder
import src.features.indicators
import src.models.cnn_lstm
importlib.reload(src.data.loader)
importlib.reload(src.labeling.oracle)
importlib.reload(src.features.builder)
importlib.reload(src.features.indicators)
importlib.reload(src.models.cnn_lstm)

from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.cnn_lstm import CNNLSTMModel

# %% [markdown]
# ## 1. Configuration
# 
# **NEW PARAMETER - LOOKBACK:**
# 
# Unlike XGBoost which sees one row at a time, CNN-LSTM sees a SEQUENCE of rows.
# `LOOKBACK=20` means the model sees the last 20 candles (5 hours for 15-min data).
# 
# This allows the model to learn patterns like:
# - "Price usually reverses after 3 consecutive red candles"
# - "High volume followed by low volume often precedes breakouts"

# %%
# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Oracle label parameters (same as baseline for fair comparison)
SIGMA = 3           # Gaussian smoothing sigma
THRESHOLD = 0.0002  # Slope threshold for direction classification

# Prediction horizon
HORIZON = 1         # Predict next bar direction

# CNN-LSTM specific parameter
LOOKBACK = 10       # How many past candles to look at (10 showed better results than 20)
                    # 20 candles × 15 min = 5 hours of history
                    # Experiment with: 5, 10, 20, 30, 50

# Train/Test split dates
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"

print("="*60)
print("📋 CONFIGURATION")
print("="*60)
print(f"Oracle: sigma={SIGMA}, threshold={THRESHOLD}")
print(f"Prediction Horizon: {HORIZON} bar(s)")
print(f"Lookback Window: {LOOKBACK} candles ({LOOKBACK * 15} minutes)")
print(f"Training: up to {TRAIN_END}")
print(f"Testing: from {TEST_START}")

# %% [markdown]
# ## 2. Load and Prepare Data
# 
# Same data loading as baseline for fair comparison.

# %%
# ==============================================================================
# LOAD DATA AND CREATE LABELS
# ==============================================================================

print("\n" + "="*60)
print("📥 LOADING DATA")
print("="*60)

df = load_and_merge_data(end_date='2025-12-31')
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)

print(f"\n✅ Loaded {len(df):,} rows")
print("\nLabel distribution:")
label_counts = df['target'].value_counts(normalize=True).sort_index()
label_names = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
for label, pct in label_counts.items():
    print(f"  {label_names[label]:8s}: {pct*100:5.1f}%")

# %%
# ==============================================================================
# GENERATE FEATURES
# ==============================================================================
#
# For neural networks, we use fewer feature groups to avoid overfitting.
# Deep learning can learn features automatically, so we don't need as many.

print("\n" + "="*60)
print("⚙️ GENERATING FEATURES")
print("="*60)

# Use fewer groups for neural network (less prone to overfitting)
feature_groups = ['momentum', 'overlap', 'trend', 'volatility', 'volume', 'statistics']
df_features, group_map = prepare_features(df, groups=feature_groups, horizon=HORIZON)

print("\nFeature groups used:")
for group, cols in group_map.items():
    print(f"  {group}: {len(cols)} features")

# %%
# ==============================================================================
# TIME-BASED SPLIT
# ==============================================================================

print("\n" + "="*60)
print("📊 SPLITTING DATA")
print("="*60)

train_df, val_df, test_df = split_data_by_time(
    df_features,
    train_end=TRAIN_END,
    test_start=TEST_START,
    val_ratio=0.1  # Same as XGB for fair comparison
)

feature_cols = get_indicator_columns(
    df_features, 
    exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
)
feature_cols = [c for c in feature_cols if c in train_df.columns]
print(f"\n✅ Features: {len(feature_cols)}")

# %%
# ==============================================================================
# EXTRACT FEATURE MATRICES
# ==============================================================================
#
# Note: The CNN-LSTM model will internally convert these 2D arrays
# to 3D sequences using the LOOKBACK parameter.
# Shape: (samples, features) → (samples, lookback, features)

print("\n" + "="*60)
print("📐 EXTRACTING FEATURE MATRICES")
print("="*60)

X_train = train_df[feature_cols].values
y_train = train_df['target'].values.astype(int)

X_val = val_df[feature_cols].values
y_val = val_df['target'].values.astype(int)

X_test = test_df[feature_cols].values
y_test = test_df['target'].values.astype(int)

# Clean inf/nan values (some indicators like EOM produce inf from division by zero)
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Training:   {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
print(f"Validation: {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
print(f"Test:       {X_test.shape[0]:,} samples × {X_test.shape[1]} features")

print(f"\n💡 After sequence creation (lookback={LOOKBACK}):")
print(f"   Input shape will be: (samples, {LOOKBACK}, {X_train.shape[1]})")

# %% [markdown]
# ## 3. Train CNN-LSTM Model
# 
# **MODEL ARCHITECTURE:**
# 
# ```
# Input (lookback × features)
#     ↓
# Conv1D (extract local patterns)
#     ↓
# MaxPooling (reduce dimensionality)
#     ↓
# LSTM (capture temporal dependencies)
#     ↓
# Dropout (prevent overfitting)
#     ↓
# Dense (classification)
#     ↓
# Output (3 classes)
# ```
# 
# **Key Parameters:**
# - `conv_filters`: Number of 1D convolution filters
# - `lstm_units`: LSTM hidden state size
# - `dropout`: Fraction of units to drop (regularization)
# - `learning_rate`: Step size for optimizer

# %%
# ==============================================================================
# HYPERPARAMETER TUNING (GRID SEARCH)
# ==============================================================================
#
# Перебираем комбинации гиперпараметров для нахождения лучшей модели.
# Каждая комбинация обучается с уменьшенным числом эпох для скорости.

print("\n" + "="*60)
print("🔧 HYPERPARAMETER TUNING")
print("="*60)

# Параметры для перебора
PARAM_GRID = {
    'conv_filters': [16,32, 64, 128],           # Фильтры Conv1D
    'dropout': [0.1, 0.2, 0.3],        # Dropout rate
    'learning_rate': [0.001,],   # Learning rate
    'lstm_units': [64, 96, 128],             # LSTM units
    'batch_size': [32, 64],             # Batch size
}

# Быстрый поиск с меньшим числом эпох
SEARCH_EPOCHS = 20
SEARCH_PATIENCE = 7

print(f"\nПараметры для перебора:")
for param, values in PARAM_GRID.items():
    print(f"   {param}: {values}")

total_combinations = 1
for values in PARAM_GRID.values():
    total_combinations *= len(values)
print(f"\nВсего комбинаций: {total_combinations}")
print(f"Эпох на комбинацию: {SEARCH_EPOCHS}")

# Запуск поиска
from itertools import product
import gc

results = []
best_val_acc = 0
best_params = {}

param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())

for i, combo in enumerate(product(*param_values)):
    params = dict(zip(param_names, combo))
    
    print(f"\n[{i+1}/{total_combinations}] Testing: {params}")
    
    # Создаём модель с этими параметрами
    test_model = CNNLSTMModel(
        n_classes=3,
        lookback=LOOKBACK,
        conv_filters=params['conv_filters'],
        lstm_units=params['lstm_units'],
        dropout=params['dropout'],
        learning_rate=params['learning_rate'],
        device='cuda',
        random_seed=42
    )
    
    # Быстрое обучение
    test_model.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_cols,
        epochs=SEARCH_EPOCHS,
        batch_size=params['batch_size'],
        patience=SEARCH_PATIENCE,
        use_class_weights=True  # Handle class imbalance (SIDEWAYS ~40%)
    )
    
    # Оценка на validation
    val_metrics = test_model.evaluate(X_val, y_val)
    val_acc = val_metrics['accuracy']
    
    results.append({**params, 'val_accuracy': val_acc})
    print(f"   Val Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params.copy()
        print(f"   ⭐ New best!")
    
    # Очистка памяти
    del test_model
    gc.collect()

# Результаты поиска
print("\n" + "="*60)
print("📊 TUNING RESULTS")
print("="*60)

import pandas as pd
results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False)
print(results_df.to_string(index=False))

print(f"\n🏆 Best parameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")
print(f"   Val Accuracy: {best_val_acc:.4f}")

# Save best params to JSON for train.py to use
import json
best_params_to_save = {
    **best_params,
    'val_accuracy': best_val_acc,
    'lookback': LOOKBACK  # Include lookback used in grid search
}
Path('models_artifacts').mkdir(exist_ok=True)
with open('models_artifacts/cnn_lstm_best_params.json', 'w') as f:
    json.dump(best_params_to_save, f, indent=2)
print(f"\n✅ Best params saved: models_artifacts/cnn_lstm_best_params.json")

# %%
# ==============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ==============================================================================

print("\n" + "="*60)
print("🚀 TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("="*60)

# Создаём финальную модель с лучшими параметрами
model = CNNLSTMModel(
    n_classes=3,
    lookback=LOOKBACK,
    conv_filters=best_params.get('conv_filters', 32),
    lstm_units=best_params.get('lstm_units', 64),
    dropout=best_params.get('dropout', 0.2),
    learning_rate=best_params.get('learning_rate', 0.001),
    device='cuda',
    random_seed=42
)

print(f"\n✅ Final model architecture:")
print(f"   Conv filters: {best_params.get('conv_filters', 32)}")
print(f"   LSTM units: {best_params.get('lstm_units', 64)}")
print(f"   Dropout: {best_params.get('dropout', 0.2)}")
print(f"   Learning rate: {best_params.get('learning_rate', 0.001)}")
print(f"   Batch size: {best_params.get('batch_size', 64)}")

# Полное обучение с лучшими параметрами
model.fit(
    X_train, y_train,
    X_val, y_val,
    feature_names=feature_cols,
    epochs=100,
    batch_size=best_params.get('batch_size', 64),
    patience=15,
    use_class_weights=True  # Handle class imbalance (SIDEWAYS ~40%)
)

print("\n✅ Training complete!")

# %% [markdown]
# ## 4. Training History Visualization
# 
#
# The training curves tell us:
# - Is the model learning? (loss decreasing)
# - Is it overfitting? (train loss << val loss)
# - When did it stop improving? (early stopping point)

# %%
# ==============================================================================
# PLOT TRAINING HISTORY
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves
# Good sign: Both curves decreasing, staying close together
# Bad sign: Train loss much lower than val loss (overfitting)
axes[0].plot(model.history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(model.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss (Cross-Entropy)', fontsize=11)
axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy curve
# Good sign: Accuracy increasing and stabilizing
axes[1].plot(model.history['val_acc'], label='Validation Accuracy', 
             linewidth=2, color='green')
axes[1].axhline(y=0.333, color='red', linestyle='--', label='Random Baseline (33.3%)')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_lstm_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze training
final_train_loss = model.history['train_loss'][-1]
final_val_loss = model.history['val_loss'][-1]
print("\n💡 TRAINING ANALYSIS:")
print(f"   Final train loss: {final_train_loss:.4f}")
print(f"   Final val loss:   {final_val_loss:.4f}")
if final_val_loss > final_train_loss * 1.5:
    print("   ⚠️ Gap suggests some overfitting - consider more dropout")
else:
    print("   ✅ Curves are close - model is not overfitting")

# %% [markdown]
# ## 5. Evaluation on Test Set

# %%
# ==============================================================================
# FINAL EVALUATION
# ==============================================================================

print("\n" + "="*60)
print("🎯 TEST SET EVALUATION")
print("="*60)

test_metrics = model.evaluate(X_test, y_test)

print(f"\nTest Set Results:")
print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']:.1%})")
print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
print(f"  F1 Macro:    {test_metrics['f1_macro']:.4f}")

print(f"\n  Random Baseline: 33.3%")
print(f"  Improvement:     {(test_metrics['accuracy'] - 0.333)*100:+.1f}%")

# %%
# ==============================================================================
# CONFUSION MATRIX
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

cm = np.array(test_metrics['confusion_matrix'])
labels = ['DOWN', 'SIDEWAYS', 'UP']

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 14})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Label', fontsize=12)
ax.set_title('Confusion Matrix - CNN-LSTM', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('cnn_lstm_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ==============================================================================
# SAVE MODEL
# ==============================================================================

print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

Path('models_artifacts').mkdir(exist_ok=True)
model.save('models_artifacts', name=f'cnn_lstm_h{HORIZON}')
print(f"✅ Model saved: models_artifacts/cnn_lstm_h{HORIZON}_model.keras")

# %% [markdown]
# ## 6. Hyperparameter Experiments
# 
# **WHY EXPERIMENT WITH LOOKBACK?**
# 
# The lookback window is one of the most important hyperparameters:
# - Too short: Model can't see enough history
# - Too long: Model may overfit to noise
# 
# We test several values to find the optimal one.

# %%
# ==============================================================================
# LOOKBACK EXPERIMENT
# ==============================================================================
#
# Test different lookback values to find optimal sequence length.
# This takes a while as we train multiple models.

print("\n" + "="*60)
print("🔬 LOOKBACK EXPERIMENTS")
print("="*60)
print("Testing different lookback values...")
print("This will train 4 models and may take 10-15 minutes.\n")

lookback_values = [5, 10, 20, 30]
lookback_results = []

for lb in lookback_values:
    print(f"\n--- Testing lookback={lb} ({lb * 15} minutes of history) ---")
    
    model_exp = CNNLSTMModel(
        n_classes=3,
        lookback=lb,
        conv_filters=16,
        lstm_units=64,
        dropout=0.5,
        device='cuda',
        random_seed=42
    )
    
    # Train with fewer epochs for experiments
    model_exp.fit(
        X_train, y_train, 
        X_val, y_val,
        epochs=50,       # Fewer epochs for speed
        batch_size=64, 
        patience=10,
        scale=True
    )
    
    metrics = model_exp.evaluate(X_test, y_test)
    
    lookback_results.append({
        'lookback': lb,
        'minutes': lb * 15,
        'accuracy': metrics['accuracy'],
        'f1_weighted': metrics['f1_weighted']
    })
    
    print(f"   Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")

# Display results
print("\n" + "="*60)
print("📊 LOOKBACK COMPARISON RESULTS")
print("="*60)

results_df = pd.DataFrame(lookback_results)
print("\n" + results_df.to_string(index=False))

# Find best
best_idx = results_df['accuracy'].idxmax()
best_lookback = results_df.loc[best_idx, 'lookback']
print(f"\n✅ Best lookback: {best_lookback} candles ({best_lookback * 15} minutes)")

# %% [markdown]
# ## 7. Summary

# %%
# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*60)
print("📋 CNN-LSTM MODEL SUMMARY")
print("="*60)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Lookback: {LOOKBACK} candles ({LOOKBACK * 15} minutes)
• Architecture: Conv1D(16) → LSTM(64) → Dense(32) → Output(3)
• Dropout: 50%
• Learning Rate: 0.001
• Optimizer: Adam

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TEST SET RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']:.1%})
• F1 Weighted: {test_metrics['f1_weighted']:.4f}
• F1 Macro:    {test_metrics['f1_macro']:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 LOOKBACK EXPERIMENT RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{results_df.to_string(index=False)}

Best: lookback={best_lookback}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💾 SAVED FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Model: models_artifacts/cnn_lstm_h{HORIZON}_model.keras
• Training plot: cnn_lstm_training_history.png
• Confusion matrix: cnn_lstm_confusion_matrix.png
""")

print("="*60)
print("✅ CNN-LSTM TRAINING COMPLETE!")
print("="*60)
print("\nNext step: Run 04_comparison.py to compare models.")
