# Capstone Project: BTC Price Direction Prediction

## Notebooks - Run in Order!

This project trains ML models to predict Bitcoin price direction.
**Run notebooks in numerical order (01 → 05).**

### Pipeline Overview

```
┌─────────────────────────────────────┐
│  01_eda.ipynb                       │  ← Explore data, create oracle labels
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  02_baseline_xgb.ipynb              │  ← Train XGBoost (horizon=1) + grid search
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  03_cnn_lstm.ipynb                  │  ← Train CNN-LSTM (horizon=1) + grid search
│                                     │     Saves: cnn_lstm_best_params.json
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  03b_tcn_attention.py               │  ← Train TCN-Attention (horizon=1) + grid search
│                                     │     Saves: tcn_attention_best_params.json
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  04_train_additional_horizons.py    │  ← Train models for h=3,5 (script)
│                                     │     Uses best_params.json from notebooks
└───────────┬─────────────────────────┘  
            ↓
┌─────────────────────────────────────┐
│  05_comparison.ipynb                │  ← Compare ALL trained models
└─────────────────────────────────────┘
```

## Alternative: Using train.py CLI

Instead of notebooks, you can use the CLI training script:

```bash
# Train XGBoost baseline
python -m src.models.train --config configs/baseline.yaml

# Train CNN-LSTM (uses cnn_lstm_best_params.json from notebook if exists)
python -m src.models.train --config configs/cnn_lstm.yaml
```

> **ВАЖНО**: train.py загружает `*_best_params.json` из `models_artifacts/` если файл существует.
> Это значит: **сначала запусти notebooks для grid search**, потом можно использовать train.py!

### Notebook Details

| # | Notebook | Purpose | Output | Time |
|---|----------|---------|--------|------|
| 01 | `01_eda.ipynb` | Data exploration, oracle labels | EDA plots | ~5 min |
| 02 | `02_baseline_xgb.ipynb` | XGBoost baseline (h=1) | `xgb_baseline_h1_*` | ~10 min |
| 03 | `03_cnn_lstm.ipynb` | CNN-LSTM + grid search (h=1) | `cnn_lstm_best_params.json`, model | ~30 min |
| 03b | `03b_tcn_attention.py` | TCN-Attention + grid search (h=1) | `tcn_attention_best_params.json`, model | ~30 min |
| 04 | `04_train_additional_horizons.py` | All models for h=3,5 | h3, h5 models | ~60 min |
| 05 | `05_comparison.ipynb` | Compare all models | comparison report | ~5 min |

### How to Run

```bash
# WSL with conda environment
cd /mnt/c/_PYTH/projects/capstone_project/notebooks
conda activate rapids-25.10

# Step 1-3: Run notebooks in Jupyter or VS Code
# 01_eda.ipynb → 02_baseline_xgb.ipynb → 03_cnn_lstm.ipynb

# Step 3b: TCN-Attention (script with grid search)
python 03b_tcn_attention.py

# Step 4: Train additional horizons (uses best_params from step 3, 3b)
python 04_train_additional_horizons.py

# Step 5: Final comparison
jupyter notebook 05_comparison.ipynb
```

### What Each Step Produces

1. **01_eda**: Visual analysis, saves EDA plots
2. **02_baseline_xgb**: Saves `models_artifacts/xgb_baseline_h1_*`
3. **03_cnn_lstm**: Saves `cnn_lstm_best_params.json` + `cnn_lstm_h1_model.keras`
4. **03b_tcn_attention**: Saves `tcn_attention_best_params.json` + `tcn_attention_h1_model.keras`
5. **04_train_additional_horizons**: Loads `*_best_params.json`, trains h3 and h5 models
6. **05_comparison**: Comparison charts, McNemar test, final report

### Class Weights (Handling Imbalance)

All neural networks use `use_class_weights=True` to handle class imbalance:
- SIDEWAYS class is ~40% of data
- Without class weights, models just predict SIDEWAYS everywhere (fake 70% accuracy!)
- F1 Macro is the real metric (should be ~0.50-0.55)

### GPU Acceleration

All training uses GPU:
- **XGBoost**: Uses CUDA via `device='cuda'`
- **CNN-LSTM**: Uses TensorFlow GPU automatically
- **TCN-Attention**: Uses TensorFlow GPU automatically

### Output Structure

```
notebooks/
├── models_artifacts/
│   ├── cnn_lstm_best_params.json      ← Grid search results
│   ├── tcn_attention_best_params.json ← Grid search results
│   ├── xgb_baseline_h1_model.joblib
│   ├── xgb_baseline_h3_model.joblib
│   ├── xgb_baseline_h5_model.joblib
│   ├── cnn_lstm_h1_model.keras
│   ├── cnn_lstm_h3_model.keras
│   ├── cnn_lstm_h5_model.keras
│   ├── tcn_attention_h1_model.keras
│   ├── tcn_attention_h3_model.keras
│   └── tcn_attention_h5_model.keras
└── reports/
    ├── model_comparison.csv
    └── comparison_summary.json
```
