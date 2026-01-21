"""
02 - Baseline XGBoost Model

This notebook trains and evaluates the XGBoost baseline model:
1. Data preparation with feature engineering
2. Walk-forward cross-validation
3. Hyperparameter tuning
4. Feature importance analysis
5. Evaluation on test set

Run this script or convert to Jupyter notebook.
"""

# %% [markdown]
# # XGBoost Baseline Model
# 
# Train XGBoost classifier for Bitcoin price direction prediction.

# %%
import sys
from pathlib import Path

project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# %%
from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.xgb import XGBBaseline, print_classification_report

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Configuration
SIGMA = 4
THRESHOLD = 0.0004
HORIZON = 1
TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"

# %%
# Load data
print("Loading data...")
df = load_and_merge_data()
print(f"Shape: {df.shape}")

# Create oracle labels
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)
print("\nLabel distribution:")
print(df['target'].value_counts(normalize=True))

# %%
# Prepare features
df_features, group_map = prepare_features(df, horizon=HORIZON)

print("\nFeatures by group:")
for group, cols in group_map.items():
    print(f"  {group}: {len(cols)} features")

# %%
# Split data
train_df, val_df, test_df = split_data_by_time(
    df_features,
    train_end=TRAIN_END,
    test_start=TEST_START,
    val_ratio=0.1
)

# Get feature columns
feature_cols = get_indicator_columns(df_features, 
                                     ['time', 'target', 'smoothed_close', 'smooth_slope'])
feature_cols = [c for c in feature_cols if c in train_df.columns]
print(f"\nTotal features: {len(feature_cols)}")

# %%
# Extract X, y
X_train = train_df[feature_cols].values
y_train = train_df['target'].values.astype(int)

X_val = val_df[feature_cols].values
y_val = val_df['target'].values.astype(int)

X_test = test_df[feature_cols].values
y_test = test_df['target'].values.astype(int)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 2. Train Baseline Model

# %%
# Create and train model
model = XGBBaseline(
    n_classes=3,
    device='cuda',
    random_state=42
)

# Fit without tuning first
model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

# Evaluate on validation
val_metrics = model.evaluate(X_val, y_val)
print_classification_report(val_metrics, "Validation Results (No Tuning)")

# %% [markdown]
# ## 3. Hyperparameter Tuning

# %%
# Tune with RandomizedSearchCV
best_params = model.tune(
    X_train, y_train,
    n_iter=25,
    cv_splits=5,
    scoring='f1_weighted'
)

# %% [markdown]
# ## 4. Final Evaluation

# %%
# Evaluate on test set
test_metrics = model.evaluate(X_test, y_test)
print_classification_report(test_metrics, "Test Results (After Tuning)")

# %%
# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))

cm = np.array(test_metrics['confusion_matrix'])
labels = ['DOWN', 'SIDEWAYS', 'UP']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - XGBoost Baseline')

plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png', dpi=150)
plt.show()

# %% [markdown]
# ## 5. Feature Importance

# %%
# Get feature importance
fi = model.get_feature_importance()

# Plot top 20 features
fig, ax = plt.subplots(figsize=(10, 8))

top_20 = fi.head(20)
ax.barh(range(len(top_20)), top_20['Importance'].values)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Top 20 Features by Importance')

plt.tight_layout()
plt.savefig('baseline_feature_importance.png', dpi=150)
plt.show()

# %%
# Save model
model.save('models_artifacts', name=f'xgb_baseline_h{HORIZON}')
fi.to_csv(f'models_artifacts/feature_importance_h{HORIZON}.csv', index=False)

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "="*60)
print("📋 BASELINE MODEL SUMMARY")
print("="*60)

print(f"""
Configuration:
- Sigma: {SIGMA}
- Threshold: {THRESHOLD}
- Horizon: {HORIZON} bar(s)

Results:
- Accuracy: {test_metrics['accuracy']:.4f}
- F1 Weighted: {test_metrics['f1_weighted']:.4f}
- F1 Macro: {test_metrics['f1_macro']:.4f}

Best Parameters:
{model.best_params}

Model saved to: models_artifacts/xgb_baseline_h{HORIZON}_model.joblib
""")
