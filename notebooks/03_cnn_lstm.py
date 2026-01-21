"""
03 - CNN-LSTM Advanced Model

This notebook trains and evaluates the CNN-LSTM model:
1. Data preparation with sequence creation
2. Model architecture and training
3. Hyperparameter experiments
4. Evaluation and comparison with baseline

Run this script or convert to Jupyter notebook.
"""

# %% [markdown]
# # CNN-LSTM Advanced Model
# 
# Train CNN-LSTM neural network for Bitcoin price direction prediction.

# %%
import sys
from pathlib import Path

project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.cnn_lstm import CNNLSTMModel

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Configuration
SIGMA = 4
THRESHOLD = 0.0004
HORIZON = 1
LOOKBACK = 20
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"

# %%
# Load data
print("Loading data...")
df = load_and_merge_data()
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)

print(f"Shape: {df.shape}")
print("\nLabel distribution:")
print(df['target'].value_counts(normalize=True))

# %%
# Prepare features (fewer groups for neural network)
feature_groups = ['momentum', 'overlap', 'trend', 'volatility', 'volume', 'statistics']
df_features, group_map = prepare_features(df, groups=feature_groups, horizon=HORIZON)

# %%
# Split data
train_df, val_df, test_df = split_data_by_time(
    df_features,
    train_end=TRAIN_END,
    test_start=TEST_START,
    val_ratio=0.15
)

feature_cols = get_indicator_columns(df_features, 
                                     ['time', 'target', 'smoothed_close', 'smooth_slope'])
feature_cols = [c for c in feature_cols if c in train_df.columns]
print(f"\nFeatures: {len(feature_cols)}")

# %%
# Extract arrays
X_train = train_df[feature_cols].values
y_train = train_df['target'].values.astype(int)

X_val = val_df[feature_cols].values
y_val = val_df['target'].values.astype(int)

X_test = test_df[feature_cols].values
y_test = test_df['target'].values.astype(int)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 2. Train CNN-LSTM Model

# %%
# Create model
model = CNNLSTMModel(
    n_classes=3,
    lookback=LOOKBACK,
    conv_filters=16,
    lstm_units=64,
    dropout=0.5,
    dense_units=32,
    learning_rate=0.001,
    device='cuda',
    random_seed=42
)

# %%
# Train
model.fit(
    X_train, y_train,
    X_val, y_val,
    feature_names=feature_cols,
    epochs=100,
    batch_size=64,
    patience=15
)

# %% [markdown]
# ## 3. Training History

# %%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(model.history['train_loss'], label='Train Loss')
axes[0].plot(model.history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()

# Accuracy
axes[1].plot(model.history['val_acc'], label='Val Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('cnn_lstm_training_history.png', dpi=150)
plt.show()

# %% [markdown]
# ## 4. Evaluation

# %%
# Evaluate on test set
test_metrics = model.evaluate(X_test, y_test)

print("\n" + "="*60)
print("📊 TEST SET RESULTS")
print("="*60)
print(f"\nAccuracy:    {test_metrics['accuracy']:.4f}")
print(f"F1 Weighted: {test_metrics['f1_weighted']:.4f}")
print(f"F1 Macro:    {test_metrics['f1_macro']:.4f}")

# %%
# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))

cm = np.array(test_metrics['confusion_matrix'])
labels = ['DOWN', 'SIDEWAYS', 'UP']

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - CNN-LSTM')

plt.tight_layout()
plt.savefig('cnn_lstm_confusion_matrix.png', dpi=150)
plt.show()

# %%
# Save model
model.save('models_artifacts', name=f'cnn_lstm_h{HORIZON}')

# %% [markdown]
# ## 5. Hyperparameter Experiments

# %%
# Test different lookback values
print("\n" + "="*60)
print("📊 LOOKBACK EXPERIMENTS")
print("="*60)

lookback_results = []

for lb in [5, 10, 20, 30]:
    print(f"\nTesting lookback={lb}...")
    
    model_exp = CNNLSTMModel(
        n_classes=3,
        lookback=lb,
        conv_filters=16,
        lstm_units=64,
        dropout=0.5,
        device='cuda',
        random_seed=42
    )
    
    model_exp.fit(X_train, y_train, X_val, y_val,
                  epochs=50, batch_size=64, patience=10, scale=True)
    
    metrics = model_exp.evaluate(X_test, y_test)
    
    lookback_results.append({
        'lookback': lb,
        'accuracy': metrics['accuracy'],
        'f1_weighted': metrics['f1_weighted']
    })

results_df = pd.DataFrame(lookback_results)
print("\nLookback comparison:")
print(results_df.to_string(index=False))

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "="*60)
print("📋 CNN-LSTM MODEL SUMMARY")
print("="*60)

print(f"""
Configuration:
- Lookback: {LOOKBACK}
- Conv Filters: 16
- LSTM Units: 64
- Dropout: 0.5
- Learning Rate: 0.001

Results:
- Accuracy: {test_metrics['accuracy']:.4f}
- F1 Weighted: {test_metrics['f1_weighted']:.4f}
- F1 Macro: {test_metrics['f1_macro']:.4f}

Model saved to: models_artifacts/cnn_lstm_h{HORIZON}_model.pt
""")
