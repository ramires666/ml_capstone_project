# Bitcoin Price Direction Predictor

A machine learning capstone project for predicting BTC price direction (UP/DOWN/SIDEWAYS) on 15-minute candles.

## 🎯 Problem Statement

Predicting exact cryptocurrency prices is nearly impossible due to high volatility. However, predicting **price direction** is more achievable and practically useful for trading decisions.

This project implements:
- **Baseline Model:** XGBoost classifier with 60+ technical indicators
- **Advanced Model:** CNN-LSTM neural network for capturing temporal patterns

## 📊 Data

- **Source:** Binance Futures (BTCUSDT)
- **Timeframe:** 15 minutes
- **Features:** OHLCV + 60+ technical indicators (momentum, trend, volatility, volume, candle patterns)
- **Target:** Oracle labels via Gaussian smoothing (3 classes: DOWN=0, SIDEWAYS=1, UP=2)

## 🏗️ Project Structure

```
├── configs/                    # Model configuration files
│   ├── baseline.yaml           # XGBoost configuration
│   └── cnn_lstm.yaml           # CNN-LSTM configuration
├── notebooks/                  # Analysis notebooks (Python scripts)
│   ├── 01_eda.py               # Exploratory Data Analysis
│   ├── 02_baseline_xgb.py      # XGBoost training
│   ├── 03_cnn_lstm.py          # CNN-LSTM training
│   └── 04_comparison.py        # Model comparison
├── scripts/                    # Utility scripts
│   └── compare_horizons.py     # Compare prediction horizons
├── src/                        # Source code
│   ├── data/                   # Data loading and processing
│   │   ├── downloader.py       # Download from Binance
│   │   ├── parser.py           # Parse ZIP to Parquet
│   │   └── loader.py           # Load and merge data
│   ├── features/               # Feature engineering
│   │   ├── indicators.py       # Technical indicators
│   │   └── builder.py          # Feature pipeline
│   ├── labeling/               # Target generation
│   │   └── oracle.py           # Oracle labels (Gaussian smoothing)
│   ├── models/                 # ML models
│   │   ├── xgb.py              # XGBoost classifier
│   │   ├── cnn_lstm.py         # CNN-LSTM (TensorFlow/Keras)
│   │   ├── train.py            # Training script
│   │   └── predict.py          # Inference script
│   └── api/                    # REST API
│       └── app.py              # FastAPI application
├── models_artifacts/           # Saved models and scalers
├── reports/                    # Metrics and plots
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Windows 10/11 with WSL2
- NVIDIA GPU with CUDA support
- Conda (Miniconda or Anaconda)

---

## 📦 Environment Setup (WSL2)

### Step 1: Open WSL Terminal

```powershell
# In Windows PowerShell or CMD
wsl
```

### Step 2: Create Conda Environment

Create a new conda environment with Python 3.11 in a custom location:

```bash
# Create environment in specific folder (replace path as needed)
conda create --prefix /mnt/w/WSL/btc python=3.11 -y

# Activate the environment
conda activate /mnt/w/WSL/btc
```

### Step 3: Install Dependencies via Conda-Forge

Install all packages through conda-forge for proper CUDA integration:

```bash
# Install TensorFlow with CUDA support + all other dependencies
conda install -c conda-forge \
    tensorflow cudatoolkit cudnn \
    pandas numpy scipy scikit-learn xgboost \
    matplotlib seaborn \
    fastapi uvicorn \
    pyyaml joblib requests tqdm numba \
    -y

# Install pandas-ta (not available in conda-forge)
pip install pandas-ta
```

### Step 4: Verify GPU Support

```bash
# Check TensorFlow GPU detection
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# Check XGBoost CUDA support
python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"
```

Expected output:
```
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
XGBoost version: 2.x.x
```

---

## 📥 Data Download

### Step 1: Download Historical Data from Binance

```bash
# Navigate to project directory
cd /mnt/c/_PYTH/projects/capstone_project

# Download klines, funding rates, and aggregated trades
python -m src.data.downloader
```

### Step 2: Parse ZIP Files to Parquet

```bash
# Convert downloaded archives to Parquet format
python -m src.data.parser
```

---

## 🏋️ Model Training

### Train XGBoost Baseline

```bash
# Train with default configuration (horizon=1 bar)
python -m src.models.train --config configs/baseline.yaml

# Train with different prediction horizon
python -m src.models.train --config configs/baseline.yaml --horizon 3
```

### Train CNN-LSTM Advanced Model

```bash
# Train CNN-LSTM model
python -m src.models.train --config configs/cnn_lstm.yaml

# With custom horizon
python -m src.models.train --config configs/cnn_lstm.yaml --horizon 5
```

### Compare Multiple Horizons

```bash
# Train and compare horizons 1, 3, and 5
python scripts/compare_horizons.py --horizons 1 3 5
```

---

## 🔮 Making Predictions

### Command Line

```bash
# Predict using XGBoost
python -m src.models.predict --model xgb --horizon 1 --latest

# Predict using CNN-LSTM
python -m src.models.predict --model cnn_lstm --horizon 1 --latest
```

### REST API

```bash
# Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# In another terminal, test the API
curl http://localhost:8000/health
```

Open http://localhost:8000 in your browser for the web interface.

---

## 🐳 Docker Deployment

### Build Container

```bash
docker build -t btc-predictor .
```

### Run Container

```bash
# Run with GPU support
docker run --gpus all -p 8000:8000 btc-predictor

# Run without GPU
docker run -p 8000:8000 btc-predictor
```

### Test API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"candles": [...], "model": "xgb"}'
```

---

## 📈 Expected Results

| Model | Accuracy | F1 Score | Horizon |
|-------|----------|----------|---------|
| XGBoost Baseline | ~45-55% | ~0.40-0.50 | 1 bar |
| CNN-LSTM | ~45-55% | ~0.40-0.50 | 1 bar |

> Note: Results depend on market conditions and data period.

---

## 🔧 Configuration

### Baseline Configuration (`configs/baseline.yaml`)

```yaml
model:
  type: xgboost
  
labeling:
  sigma: 4           # Gaussian smoothing parameter
  threshold: 0.0004  # UP/DOWN threshold

features:
  horizon: 1         # Bars ahead to predict
  top_k_features: 30 # Use top K features

xgboost:
  device: cuda       # Use GPU
  n_estimators: 100
  max_depth: 5
```

### CNN-LSTM Configuration (`configs/cnn_lstm.yaml`)

```yaml
model:
  type: cnn_lstm

architecture:
  lookback: 20       # Sequence length
  conv_filters: 16
  lstm_units: 64
  dropout: 0.5

training:
  epochs: 100
  batch_size: 64
  patience: 15       # Early stopping
  device: cuda
```

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/docs` | GET | API documentation (Swagger) |

---

## 🧪 Running Notebooks

The notebooks are Python scripts compatible with VS Code and Jupyter:

```bash
# Option 1: Run as Python script
python notebooks/01_eda.py

# Option 2: Convert to Jupyter notebook
pip install jupytext
jupytext --to notebook notebooks/01_eda.py

# Option 3: Open in VS Code with Python extension
# Just open the .py file and run cells with # %% markers
```

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- Baseline approach from course notebook
- CNN-LSTM architecture inspired by "Bitcoin price direction prediction using on-chain data" paper
