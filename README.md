# Bitcoin Price Direction Predictor

A machine learning capstone project for predicting BTC price direction (UP/DOWN/SIDEWAYS) on 15-minute candles.

## 🎯 Problem Statement

Predicting exact cryptocurrency prices is nearly impossible due to high volatility. However, predicting **price direction** is more achievable and practically useful for trading decisions.

This project implements:

- **XGBoost Baseline:** Gradient boosting with 60+ technical indicators
- **CNN-LSTM:** Convolutional + LSTM neural network for temporal patterns
- **TCN-Attention:** Temporal Convolutional Network with Multi-Head Self-Attention

## 📊 Data

- **Source:** Binance Futures (BTCUSDT)
- **Timeframe:** 15 minutes
- **Features:** OHLCV + 60+ technical indicators (momentum, trend, volatility, volume, candle patterns)
- **Target:** Oracle labels via Gaussian smoothing (3 classes: DOWN=0, SIDEWAYS=1, UP=2)

---

## 🏗️ Project Structure

```
capstone_project/
├── configs/                        # Model configuration files
│   ├── baseline.yaml               # XGBoost configuration
│   └── cnn_lstm.yaml               # CNN-LSTM configuration
│
├── notebooks/                      # 🔬 DEVELOPMENT: Grid search & experiments
│   ├── README.md                   # ← Notebook run order documentation
│   ├── 01_eda.py                   # Exploratory Data Analysis
│   ├── 02_baseline_xgb.py          # XGBoost training + tuning
│   ├── 03_cnn_lstm.py              # CNN-LSTM grid search → cnn_lstm_best_params.json
│   ├── 03b_tcn_attention.py        # TCN-Attention grid search → tcn_attention_best_params.json
│   ├── 04_train_additional_horizons.py  # Train h=3,5 using best_params
│   ├── 05_comparison.py            # Compare all models
│   └── models_artifacts/           # Saved models from notebooks
│       ├── cnn_lstm_best_params.json
│       ├── tcn_attention_best_params.json
│       └── *.keras, *.joblib
│
├── src/                            # 🏭 PRODUCTION: Source code
│   ├── data/                       # Data loading and processing
│   │   ├── downloader.py           # Download from Binance
│   │   ├── parser.py               # Parse ZIP to Parquet
│   │   └── loader.py               # Load and merge data
│   ├── features/                   # Feature engineering
│   │   ├── indicators.py           # Technical indicators (pandas-ta)
│   │   └── builder.py              # Feature pipeline
│   ├── labeling/                   # Target generation
│   │   └── oracle.py               # Oracle labels (Gaussian smoothing)
│   ├── models/                     # ML models
│   │   ├── xgb.py                  # XGBoost classifier
│   │   ├── cnn_lstm.py             # CNN-LSTM (TensorFlow/Keras)
│   │   ├── tcn_attention.py        # TCN-Attention (TensorFlow/Keras)
│   │   ├── train.py                # CLI training script (loads best_params.json!)
│   │   └── predict.py              # Inference script
│   └── api/                        # REST API
│       └── app.py                  # FastAPI application
│
├── models_artifacts/               # Production models (from train.py)
├── reports/                        # Metrics and plots
├── Dockerfile                      # Container for production
└── requirements.txt
```

---

## 🔄 Two Workflows: Development vs Production

### 🔬 Development (Notebooks)
**Purpose:** Grid search, experiments, model research

```bash
cd notebooks/
python 01_eda.py            # EDA
python 02_baseline_xgb.py   # XGBoost tuning
python 03_cnn_lstm.py       # CNN-LSTM grid search → saves best_params.json
python 03b_tcn_attention.py # TCN-Attention grid search → saves best_params.json
python 04_train_additional_horizons.py  # Train models for h=3,5
python 05_comparison.py     # Compare all models
```

> **Output:** `notebooks/models_artifacts/cnn_lstm_best_params.json`, `.keras` models

### 🏭 Production (train.py CLI)
**Purpose:** Final training with optimal hyperparameters

```bash
# Loads best_params.json from notebooks if it exists!
python -m src.models.train --config configs/cnn_lstm.yaml

# Or XGBoost
python -m src.models.train --config configs/baseline.yaml
```

> **Important:** train.py automatically loads `cnn_lstm_best_params.json` if the file exists in `models_artifacts/`. This ensures production uses the same hyperparameters as notebook experiments.

### 🐳 Docker Deployment
**Purpose:** Run API in production

```bash
# Build
docker build -t btc-predictor .

# Run with GPU
docker run --gpus all -p 8000:8000 btc-predictor

# Test
curl http://localhost:8000/health
```

---

## 🚀 Quick Start

### Prerequisites

- Windows 10/11 with WSL2
- NVIDIA GPU with CUDA support
- Conda (Miniconda or Anaconda)

### Step 1: Environment Setup

```bash
wsl
conda activate rapids-25.10  # or your environment
cd /mnt/c/_PYTH/projects/capstone_project
```

### Step 2: Download Data

```bash
python -m src.data.downloader  # Download from Binance
python -m src.data.parser      # Convert to Parquet
```

### Step 3: Run Notebooks (Development)

```bash
cd notebooks/
# Run in order: 01 → 02 → 03 → 03b → 04 → 05
# See notebooks/README.md for details
```

### Step 4: Production Training (Optional)

```bash
# Uses best_params.json from notebooks!
python -m src.models.train --config configs/cnn_lstm.yaml
```

### Step 5: Start API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## ⚖️ Class Weights (Important!)

All neural networks use `use_class_weights=True` to handle class imbalance:

| Class | Distribution | Without Weights | With Weights |
|-------|--------------|-----------------|--------------|
| DOWN (0) | ~30% | Ignored | Balanced |
| SIDEWAYS (1) | ~40% | Over-predicted | Balanced |
| UP (2) | ~30% | Ignored | Balanced |

**Result:** F1 Macro ~0.50-0.55 instead of fake 70% accuracy

---

## 📈 Expected Results

| Model | Accuracy | F1 Macro | F1 Weighted | Notes |
|-------|----------|----------|-------------|-------|
| XGBoost | ~50-55% | ~0.45-0.50 | ~0.50-0.55 | Baseline |
| CNN-LSTM | ~50-55% | ~0.45-0.55 | ~0.50-0.55 | With class weights |
| TCN-Attention | ~50-55% | ~0.50-0.55 | ~0.52-0.55 | Best for sequences |

> **F1 Macro** is the main metric! Accuracy is misleading due to class imbalance.

---

## 🔧 Configuration

### `configs/baseline.yaml` (XGBoost)

```yaml
labeling:
  sigma: 4
  threshold: 0.0002  # UP/DOWN threshold

xgboost:
  device: cuda
  n_estimators: 100
```

### `configs/cnn_lstm.yaml` (Neural Network)

```yaml
features:
  lookback: 10  # From grid search

architecture:
  conv_filters: 64
  lstm_units: 64
  dropout: 0.3

training:
  epochs: 30
  patience: 5
  learning_rate: 0.0007
```

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/docs` | GET | Swagger docs |

---

## 📄 License

MIT License
