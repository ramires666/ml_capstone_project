"""
FastAPI Application for Bitcoin Price Direction Prediction

Endpoints:
- GET /health - Health check
- POST /predict - Make prediction for new data
- GET /latest - Get latest prediction

Based on old_project_files/app.py with improvements.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.predict import Predictor


# --- Configuration ---
MODEL_DIR = project_root / "models_artifacts"
DEFAULT_MODEL = "xgb"
DEFAULT_HORIZON = 1

# --- FastAPI App ---
app = FastAPI(
    title="Bitcoin Price Direction Predictor",
    description="Predict BTC price direction (UP/DOWN/SIDEWAYS) using ML models",
    version="1.0.0"
)

# Global predictor instances
predictors: Dict[str, Predictor] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize predictors on startup."""
    global predictors
    
    try:
        # Try to load XGBoost model
        predictors['xgb'] = Predictor(
            model_type='xgb',
            model_dir=str(MODEL_DIR),
            horizon=DEFAULT_HORIZON
        )
    except Exception as e:
        print(f"⚠️ XGBoost model not loaded: {e}")
    
    try:
        # Try to load CNN-LSTM model
        predictors['cnn_lstm'] = Predictor(
            model_type='cnn_lstm',
            model_dir=str(MODEL_DIR),
            horizon=DEFAULT_HORIZON,
            device='cuda'
        )
    except Exception as e:
        print(f"⚠️ CNN-LSTM model not loaded: {e}")
    
    if not predictors:
        print("❌ No models loaded! Train models first.")
    else:
        print(f"✅ Loaded models: {list(predictors.keys())}")


# --- Request/Response Models ---
class Candle(BaseModel):
    time: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    candles: List[Candle]
    model: str = "xgb"


class PredictResponse(BaseModel):
    prediction: int
    label: str
    icon: str
    probabilities: Dict[str, float]
    timestamp: str
    model: str
    horizon: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    timestamp: str


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": list(predictors.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make prediction for provided candles.
    
    Requires at least 200 candles for indicator calculation.
    """
    if request.model not in predictors:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not loaded. Available: {list(predictors.keys())}"
        )
    
    if len(request.candles) < 200:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 200 candles, got {len(request.candles)}"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame([c.dict() for c in request.candles])
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Make prediction
    predictor = predictors[request.model]
    result = predictor.predict(df)
    
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])
    
    return result


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML UI."""
    models_html = ", ".join(predictors.keys()) if predictors else "None (train models first)"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Price Predictor</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #eee;
                min-height: 100vh;
            }}
            h1 {{
                background: linear-gradient(90deg, #f39c12, #e74c3c);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5em;
            }}
            .card {{
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 24px;
                margin: 20px 0;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .endpoint {{
                background: rgba(52, 152, 219, 0.2);
                padding: 12px 16px;
                border-radius: 8px;
                margin: 10px 0;
                font-family: monospace;
            }}
            .method {{
                font-weight: bold;
                color: #2ecc71;
            }}
            .status {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.9em;
            }}
            .status.ok {{
                background: rgba(46, 204, 113, 0.3);
                color: #2ecc71;
            }}
        </style>
    </head>
    <body>
        <h1>₿ Bitcoin Price Direction Predictor</h1>
        
        <div class="card">
            <h2>Status</h2>
            <p>Models loaded: <span class="status ok">{models_html}</span></p>
        </div>
        
        <div class="card">
            <h2>API Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span> /health - Health check
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /predict - Make prediction
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /docs - API documentation
            </div>
        </div>
        
        <div class="card">
            <h2>Example Request</h2>
            <pre style="background: rgba(0,0,0,0.3); padding: 16px; border-radius: 8px; overflow-x: auto;">
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"candles": [...], "model": "xgb"}}'
            </pre>
        </div>
    </body>
    </html>
    """


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
