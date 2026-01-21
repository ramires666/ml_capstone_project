# Bitcoin Price Direction Predictor - Capstone Project

Предсказание направления цены BTC (UP/DOWN/SIDEWAYS) на следующих 15-минутных свечках.

## 🎯 Проблема

Предсказание точного значения цены криптовалюты практически невозможно из-за высокой волатильности и случайности. Однако предсказание **направления** движения цены (вверх/вниз/боковик) более достижимо и практически полезно для трейдинга.

**Этот проект решает задачу классификации направления цены BTC** используя:
- **Baseline:** XGBoost классификатор с техническими индикаторами и oracle labels
- **Advanced:** CNN-LSTM нейросеть для захвата временных паттернов

## 📊 Данные

- **Источник:** Binance Futures (BTCUSDT)
- **Таймфрейм:** 15 минут
- **Фичи:** OHLCV + 60+ технических индикаторов (momentum, trend, volatility, volume, candle patterns)
- **Таргет:** Oracle labels через Gaussian smoothing (3 класса: DOWN=0, SIDEWAYS=1, UP=2)

## 🏗️ Структура проекта

```
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_baseline_xgb.ipynb  # XGBoost baseline модель
│   ├── 03_cnn_lstm.ipynb      # CNN-LSTM advanced модель
│   └── 04_comparison.ipynb    # Сравнение моделей
├── src/
│   ├── data/                  # Загрузка и обработка данных
│   ├── features/              # Feature engineering
│   ├── labeling/              # Oracle labels генерация
│   ├── models/                # Модели и обучение
│   └── api/                   # FastAPI сервис
├── configs/                   # Конфигурации моделей
├── models_artifacts/          # Сохранённые модели
└── reports/                   # Метрики и графики
```

## 🚀 Quick Start

### 1. Установка зависимостей
```bash
conda create -n btc-predictor python=3.10
conda activate btc-predictor
pip install -r requirements.txt
```

### 2. Скачивание данных
```bash
python -m src.data.downloader
```

### 3. Обучение моделей
```bash
# Baseline XGBoost
python -m src.models.train --config configs/baseline.yaml

# Advanced CNN-LSTM
python -m src.models.train --config configs/cnn_lstm.yaml
```

### 4. Запуск API
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 5. Docker
```bash
docker build -t btc-predictor .
docker run -p 8000:8000 btc-predictor
```

## 📈 Результаты

| Модель | Accuracy | Macro-F1 | Horizon |
|--------|----------|----------|---------|
| XGBoost Baseline | TBD | TBD | 1 bar |
| CNN-LSTM | TBD | TBD | 1 bar |

## 🔧 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Предсказание для новых данных
- `POST /backtest` - Бэктест на исторических данных

## 📝 License

MIT
