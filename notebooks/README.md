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
│  02_baseline_xgb.ipynb              │  ← Train XGBoost (horizon=1)
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  03_cnn_lstm.ipynb                  │  ← Train CNN-LSTM (horizon=1)
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│  04_train_additional_horizons.py    │  ← Train models for h=3,5 (script)
└───────────┬─────────────────────────┘  
            ↓
┌─────────────────────────────────────┐
│  05_comparison.ipynb                │  ← Compare ALL trained models
└─────────────────────────────────────┘
```

### Notebook Details

| # | Notebook | Purpose | Time |
|---|----------|---------|------|
| 01 | `01_eda.ipynb` | Data exploration, oracle label creation | ~5 min |
| 02 | `02_baseline_xgb.ipynb` | Train XGBoost baseline model (h=1) | ~10 min |
| 03 | `03_cnn_lstm.ipynb` | Train CNN-LSTM model (h=1) | ~20 min |
| 04 | `04_train_additional_horizons.py` | Train XGBoost + CNN-LSTM for h=3,5 (**script**) | ~60 min |
| 05 | `05_comparison.ipynb` | Compare all models, statistical tests | ~5 min |

### How to Run

```bash
# WSL with conda environment
cd /mnt/c/_PYTH/projects/capstone_project/notebooks
conda activate btc

# For notebooks - open in Jupyter or VS Code
# Run 01 → 02 → 03 in order

# For step 04 (training script - takes ~60 min)
python 04_train_additional_horizons.py

# For step 05 (comparison notebook)
jupyter notebook 05_comparison.ipynb
```

### What Each Step Produces

1. **01_eda**: Visual analysis, saves EDA plots
2. **02_baseline_xgb**: Saves `models_artifacts/xgb_baseline_h1_*`
3. **03_cnn_lstm**: Saves `models_artifacts/cnn_lstm_h1_*`
4. **04_train_additional_horizons**: Saves h3 and h5 models
5. **05_comparison**: Comparison charts, McNemar test, final report

### GPU Acceleration

All training uses GPU:
- **XGBoost**: Uses CUDA via `device='cuda'`
- **CNN-LSTM**: Uses TensorFlow GPU automatically

### Output Structure

```
notebooks/
├── models_artifacts/
│   ├── xgb_baseline_h1_model.joblib
│   ├── xgb_baseline_h3_model.joblib
│   ├── xgb_baseline_h5_model.joblib
│   ├── cnn_lstm_h1_model.keras
│   ├── cnn_lstm_h3_model.keras
│   └── cnn_lstm_h5_model.keras
└── reports/
    ├── model_comparison.csv
    └── comparison_summary.json
```
