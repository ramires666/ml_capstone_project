"""
==============================================================================
04 - TRAIN MODELS FOR ADDITIONAL HORIZONS (3 and 5 bars ahead)
==============================================================================

PURPOSE OF THIS SCRIPT:
-----------------------
In the 04_comparison notebook, we discovered that models for horizons 3 and 5
were missing. This script trains those models so we can compare performance
across different prediction horizons.

WHY DIFFERENT HORIZONS MATTER:
------------------------------
- Horizon=1: Predict price direction 15 minutes ahead (1 bar)
- Horizon=3: Predict price direction 45 minutes ahead (3 bars)  
- Horizon=5: Predict price direction 75 minutes ahead (5 bars)

Generally, longer horizons are HARDER to predict because more things can 
happen in a longer time window. But sometimes longer trends are more stable
than short-term noise.

HOW TO RUN THIS SCRIPT:
-----------------------
Open a terminal in WSL with conda environment activated:

    cd /mnt/c/_PYTH/projects/capstone_project/notebooks
    conda activate btc
    python train_horizons_3_5.py

This will train 4 models (2 horizons × 2 model types) and save them to:
    models_artifacts/xgb_baseline_h3_model.joblib
    models_artifacts/xgb_baseline_h5_model.joblib
    models_artifacts/cnn_lstm_h3_model.keras
    models_artifacts/cnn_lstm_h5_model.keras

ESTIMATED TIME:
---------------
- XGBoost: ~5-10 minutes per horizon (includes hyperparameter tuning)
- CNN-LSTM: ~10-20 minutes per horizon (depends on GPU speed)
- Total: approximately 30-60 minutes for all 4 models

AFTER RUNNING THIS SCRIPT:
--------------------------
Re-run the 04_comparison notebook to see the full horizon comparison table.
"""

# ==============================================================================
# STEP 0: IMPORTS AND SETUP
# ==============================================================================
#
# We need to import all the same modules we use in the training notebooks.
# This ensures consistency between notebook training and script training.

import sys
from pathlib import Path

# Add the project root directory to Python's path.
# This allows us to import our custom modules from the src/ folder.
# We go up one level from notebooks/ to capstone_project/
project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

# Standard data science imports
import numpy as np
import json
import warnings

# Suppress warnings to keep output clean during training.
# In production, you might want to log these instead.
warnings.filterwarnings('ignore')

# Import our custom modules from the src/ directory.
# These are the same modules used in notebooks 01-04.
from src.data.loader import load_and_merge_data, split_data_by_time
from src.labeling.oracle import create_oracle_labels
from src.features.builder import prepare_features
from src.features.indicators import get_indicator_columns
from src.models.xgb import XGBBaseline
from src.models.cnn_lstm import CNNLSTMModel

print("=" * 70)
print("🎓 TRAINING MODELS FOR HORIZONS 3 AND 5")
print("=" * 70)
print("""
This script will train XGBoost and CNN-LSTM models for two prediction horizons:
  - Horizon 3 = predict direction 45 minutes ahead (3 × 15-min bars)
  - Horizon 5 = predict direction 75 minutes ahead (5 × 15-min bars)

Each horizon requires training 2 models = 4 models total.
This will take approximately 30-60 minutes depending on your GPU.
""")

# ==============================================================================
# STEP 1: CONFIGURATION
# ==============================================================================
#
# All the key parameters are defined here in one place.
# This makes it easy to experiment with different settings.

# Which horizons we want to train models for.
# Horizon=1 is already done in notebook 02 and 03.
# We need horizons 3 and 5 to complete the comparison.
HORIZONS_TO_TRAIN = [3, 5]

# Oracle label parameters - these MUST match what we used in notebooks 01-03.
# If you use different values here, the models won't be comparable!
SIGMA = 4           # Gaussian smoothing strength (higher = smoother trends)
THRESHOLD = 0.0002  # Minimum slope to classify as UP or DOWN

# Train/Test split dates - MUST match the original training notebooks.
# This ensures all models are evaluated on the exact same test data.
TRAIN_END = "2025-06-30"    # Last date used for training
TEST_START = "2025-07-01"   # First date used for testing

# Output directory for saved models
MODEL_DIR = 'models_artifacts'

print("📋 Configuration:")
print(f"   Horizons to train: {HORIZONS_TO_TRAIN}")
print(f"   Oracle: sigma={SIGMA}, threshold={THRESHOLD}")
print(f"   Train period: up to {TRAIN_END}")
print(f"   Test period: from {TEST_START}")
print()

# ==============================================================================
# STEP 2: LOAD BASE DATA (DONE ONCE, USED FOR ALL HORIZONS)
# ==============================================================================
#
# Loading and labeling data is the same regardless of the horizon.
# We do this ONCE here and then reuse for each horizon.
# This saves time compared to loading data inside the loop.

print("=" * 70)
print("📥 STEP 2: LOADING AND LABELING DATA")
print("=" * 70)

# Load the raw data (OHLCV candles + funding rates + volume breakdown).
# This combines data from multiple parquet files into one DataFrame.
print("\n  [1/2] Loading raw data from parquet files...")
df = load_and_merge_data(end_date='2025-12-31')
print(f"        ✓ Loaded {len(df):,} rows of 15-minute candles")

# Create oracle labels (the target variable we're trying to predict).
# Oracle uses Gaussian smoothing to create cleaner UP/DOWN/SIDEWAYS labels.
print("\n  [2/2] Creating oracle labels with Gaussian smoothing...")
df = create_oracle_labels(df, sigma=SIGMA, threshold=THRESHOLD)
print(f"        ✓ Labels created with sigma={SIGMA}, threshold={THRESHOLD}")

# Show label distribution to verify it looks reasonable.
label_dist = df['target'].value_counts(normalize=True).sort_index()
label_names = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
print("\n  Label distribution:")
for label, pct in label_dist.items():
    print(f"        {label_names[label]:8s}: {pct*100:5.1f}%")

print("\n✅ Data loading complete!")

# ==============================================================================
# STEP 3: TRAIN MODELS FOR EACH HORIZON
# ==============================================================================
#
# Now we loop through each horizon and train both XGBoost and CNN-LSTM.
# The key difference for each horizon is how far ahead we shift the target.

# This dictionary will store results for summary at the end.
all_results = {}

# Process each horizon one at a time.
for horizon_index, horizon in enumerate(HORIZONS_TO_TRAIN, start=1):
    
    print("\n" + "=" * 70)
    print(f"🎯 HORIZON {horizon_index}/{len(HORIZONS_TO_TRAIN)}: Training for horizon={horizon} ({horizon * 15} minutes ahead)")
    print("=" * 70)
    
    all_results[horizon] = {}
    
    # =========================================================================
    # STEP 3a: PREPARE FEATURES FOR THIS HORIZON
    # =========================================================================
    #
    # The prepare_features() function does two things:
    # 1. Adds 60+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
    # 2. Shifts the target column by 'horizon' rows to prevent data leakage
    #
    # The shift is crucial: when predicting horizon=3, we shift target by 3
    # so that each row's features predict the direction 3 bars later.
    
    print(f"\n  📊 Preparing features with horizon={horizon} shift...")
    print(f"      This means: features at time T will predict direction at time T+{horizon}")
    
    # IMPORTANT: We use df.copy() to avoid modifying the original DataFrame.
    # Each horizon needs its own feature preparation because of different shifts.
    df_features, group_map = prepare_features(df.copy(), horizon=horizon)
    
    # Count how many features we generated
    total_features = sum(len(cols) for cols in group_map.values())
    print(f"      ✓ Generated {total_features} features across {len(group_map)} groups")
    
    # =========================================================================
    # STEP 3b: SPLIT DATA INTO TRAIN / VALIDATION / TEST SETS
    # =========================================================================
    #
    # For time series, we MUST use time-based splitting (not random!).
    # The future data must never be seen during training.
    #
    # Timeline:
    #   [Training Data]  |  [Validation]  |  [Test Data]
    #   Jan 2024 ------> |  ~Jun 2025     |  Jul 2025 -->
    
    print(f"\n  📊 Splitting data by time...")
    
    train_df, val_df, test_df = split_data_by_time(
        df_features,
        train_end=TRAIN_END,
        test_start=TEST_START,
        val_ratio=0.1  # Last 10% of training period is used for validation
    )
    
    print(f"      ✓ Training samples:   {len(train_df):,}")
    print(f"      ✓ Validation samples: {len(val_df):,}")
    print(f"      ✓ Test samples:       {len(test_df):,}")
    
    # =========================================================================
    # STEP 3c: EXTRACT FEATURE COLUMNS AND CONVERT TO NUMPY ARRAYS
    # =========================================================================
    #
    # We need to extract just the feature columns (not target, not metadata).
    # Then convert to numpy arrays for scikit-learn and TensorFlow.
    
    print(f"\n  📊 Extracting feature matrices...")
    
    # Get list of feature column names (excludes target and helper columns)
    feature_cols = get_indicator_columns(
        df_features, 
        exclude_cols=['time', 'target', 'smoothed_close', 'smooth_slope']
    )
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    print(f"      ✓ Using {len(feature_cols)} features for training")
    
    # Convert to numpy arrays for the models.
    # nan_to_num replaces NaN, +inf, -inf with 0 to prevent training errors.
    X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train_df['target'].values.astype(int)
    
    X_val = np.nan_to_num(val_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_val = val_df['target'].values.astype(int)
    
    X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = test_df['target'].values.astype(int)
    
    print(f"      ✓ Train shape: {X_train.shape}")
    print(f"      ✓ Val shape:   {X_val.shape}")
    print(f"      ✓ Test shape:  {X_test.shape}")
    
    # =========================================================================
    # STEP 3d: TRAIN XGBOOST MODEL
    # =========================================================================
    #
    # XGBoost is our baseline model. It's fast, interpretable, and works
    # well on tabular data. We use GPU acceleration for faster training.
    
    print(f"\n  {'─' * 60}")
    print(f"  🌲 TRAINING XGBOOST (Model 1/2 for horizon={horizon})")
    print(f"  {'─' * 60}")
    
    # Create the XGBoost model wrapper.
    # n_classes=3 because we have 3 labels: DOWN(0), SIDEWAYS(1), UP(2)
    # device='cuda' enables GPU training which is much faster
    print(f"\n      [1/3] Initializing XGBoost classifier...")
    xgb_model = XGBBaseline(
        n_classes=3,        # 3-class classification
        device='cuda',      # Use GPU for training
        random_state=42     # For reproducibility
    )
    
    # Train the model on training data with validation for early stopping.
    # Early stopping prevents overfitting by stopping when validation score stops improving.
    print(f"      [2/3] Training with early stopping...")
    xgb_model.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_cols  # Useful for feature importance later
    )
    
    # Hyperparameter tuning with RandomizedSearchCV.
    # This tries different combinations of parameters to find the best ones.
    # We use fewer iterations here (15 instead of 25) for faster execution.
    print(f"      [3/3] Tuning hyperparameters (this may take 5-10 minutes)...")
    best_params = xgb_model.tune(
        X_train, y_train,
        n_iter=15,              # Try 15 random parameter combinations
        cv_splits=3,            # 3-fold time series cross-validation
        scoring='f1_weighted'   # Optimize for weighted F1 score
    )
    
    print(f"\n      Best parameters found:")
    for param, value in best_params.items():
        print(f"        {param}: {value}")
    
    # Evaluate on the held-out test set.
    # This gives us an unbiased estimate of real-world performance.
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    
    print(f"\n      ✅ XGBoost Results (horizon={horizon}):")
    print(f"         Accuracy:    {xgb_metrics['accuracy']:.4f} ({xgb_metrics['accuracy']:.1%})")
    print(f"         F1 Weighted: {xgb_metrics['f1_weighted']:.4f}")
    print(f"         F1 Macro:    {xgb_metrics['f1_macro']:.4f}")
    
    # Save the trained model to disk.
    Path(MODEL_DIR).mkdir(exist_ok=True)
    xgb_model.save(MODEL_DIR, name=f'xgb_baseline_h{horizon}')
    print(f"\n      💾 Model saved: {MODEL_DIR}/xgb_baseline_h{horizon}_model.joblib")
    
    # Store results for summary table
    all_results[horizon]['XGBoost'] = xgb_metrics
    
    # =========================================================================
    # STEP 3e: TRAIN CNN-LSTM MODEL
    # =========================================================================
    #
    # CNN-LSTM is our advanced deep learning model. It can capture both
    # local patterns (CNN) and temporal dependencies (LSTM).
    #
    # CNN = Convolutional Neural Network - good at finding local patterns
    # LSTM = Long Short-Term Memory - good at remembering past information
    
    print(f"\n  {'─' * 60}")
    print(f"  🧠 TRAINING CNN-LSTM (Model 2/2 for horizon={horizon})")
    print(f"  {'─' * 60}")
    
    # Create the CNN-LSTM model.
    # These hyperparameters were tuned in notebook 03 for horizon=1.
    # We reuse them here as a reasonable starting point.
    print(f"\n      [1/2] Initializing CNN-LSTM model...")
    cnn_model = CNNLSTMModel(
        n_classes=3,                    # 3-class classification
        lookback=32,                    # Look at 32 time steps of history
        conv_filters=64,                # 64 convolutional filters
        lstm_units=64,                  # 64 LSTM units
        dropout=0.3,                    # 30% dropout to prevent overfitting
        learning_rate=0.0007,           # Adam learning rate (reduced 30% from 0.001 for stability)
        device='cuda'                   # Use GPU for training
    )
    
    # Train the model.
    # IMPORTANT: We use aggressive early stopping to prevent overfitting.
    # Based on observed training behavior:
    #   - Accuracy typically peaks in first 3-5 epochs
    #   - Then slowly decreases over next 10+ epochs if we keep training
    #   - This is classic overfitting - model memorizes training data
    #
    # Our solution:
    #   - patience=5: Stop if no improvement for 5 epochs (was 10)
    #   - epochs=30: Maximum epochs (was 50), but early stopping usually kicks in earlier
    #   - batch_size=128: Good balance between speed and gradient quality
    print(f"      [2/2] Training neural network...")
    print(f"           Using aggressive early stopping (patience=5) to prevent overfitting")
    print(f"           Watch for training progress below...")
    
    history = cnn_model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=30,                      # Maximum epochs (reduced from 50)
        batch_size=128,                 # Samples per gradient update
        patience=5                      # Stop if no improvement for 5 epochs
    )
    
    # Evaluate on test set
    cnn_metrics = cnn_model.evaluate(X_test, y_test)
    
    print(f"\n      ✅ CNN-LSTM Results (horizon={horizon}):")
    print(f"         Accuracy:    {cnn_metrics['accuracy']:.4f} ({cnn_metrics['accuracy']:.1%})")
    print(f"         F1 Weighted: {cnn_metrics['f1_weighted']:.4f}")
    print(f"         F1 Macro:    {cnn_metrics['f1_macro']:.4f}")
    
    # Save the trained model
    cnn_model.save(MODEL_DIR, name=f'cnn_lstm_h{horizon}')
    print(f"\n      💾 Model saved: {MODEL_DIR}/cnn_lstm_h{horizon}_model.keras")
    
    # Store results
    all_results[horizon]['CNN-LSTM'] = cnn_metrics
    
    print(f"\n  ✅ Horizon={horizon} complete! Both models trained and saved.")

# ==============================================================================
# STEP 4: FINAL SUMMARY
# ==============================================================================
#
# Print a summary of all trained models so you can see results at a glance.

print("\n" + "=" * 70)
print("📋 TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
{'Horizon':<12} {'Model':<12} {'Accuracy':<12} {'F1 Weighted':<12}
{'─' * 48}""")

for horizon in HORIZONS_TO_TRAIN:
    for model_name in ['XGBoost', 'CNN-LSTM']:
        metrics = all_results[horizon][model_name]
        print(f"{horizon:<12} {model_name:<12} {metrics['accuracy']:.4f}       {metrics['f1_weighted']:.4f}")

print(f"""
{'─' * 48}

📁 Models saved to: {MODEL_DIR}/
   - xgb_baseline_h3_model.joblib
   - xgb_baseline_h5_model.joblib
   - cnn_lstm_h3_model.keras
   - cnn_lstm_h5_model.keras

🔜 NEXT STEP:
   Re-run notebook 04_comparison.ipynb to see the full comparison
   across all horizons (1, 3, and 5).

✅ ALL DONE!
""")
