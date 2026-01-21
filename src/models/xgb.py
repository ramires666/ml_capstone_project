"""
XGBoost Baseline Model for Bitcoin Price Prediction

Based on old_project_files/train.py with improvements.
Uses CUDA for GPU acceleration.
"""

import os
# Workaround for Windows OpenMP/scikit-learn conflict
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score)
from sklearn.preprocessing import RobustScaler
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class XGBBaseline:
    """
    XGBoost Baseline classifier for price direction prediction.
    
    Features:
    - GPU acceleration via device='cuda'
    - Walk-forward cross-validation with TimeSeriesSplit
    - Hyperparameter tuning with RandomizedSearchCV
    - Feature importance analysis
    - Save/load functionality
    """
    
    def __init__(self, 
                 n_classes: int = 3,
                 device: str = 'cuda',
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_classes: Number of classes (2 or 3)
            device: 'cuda' or 'cpu'
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBClassifier parameters
        """
        self.n_classes = n_classes
        self.device = device
        self.random_state = random_state
        
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'device': device,
            'random_state': random_state,
        }
        default_params.update(kwargs)
        
        self.model = XGBClassifier(**default_params)
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.best_params: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.is_fitted = False
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None,
            scale: bool = True) -> 'XGBBaseline':
        """
        Fit the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names for interpretability
            scale: Whether to scale features
        
        Returns:
            self
        """
        if feature_names:
            self.feature_names = feature_names
        
        # Scale features
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Fit model
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train, verbose=False)
        
        self.is_fitted = True
        
        return self
    
    def tune(self,
             X: np.ndarray,
             y: np.ndarray,
             param_dist: Optional[Dict] = None,
             n_iter: int = 25,
             cv_splits: int = 5,
             scoring: str = 'f1_weighted',
             scale: bool = True) -> Dict[str, Any]:
        """
        Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit.
        
        Args:
            X: Features
            y: Labels
            param_dist: Parameter distribution for search
            n_iter: Number of random combinations to try
            cv_splits: Number of cross-validation splits
            scoring: Scoring metric
            scale: Whether to scale features
        
        Returns:
            Best parameters dict
        """
        print("\n🔧 Tuning hyperparameters with TimeSeriesSplit...")
        
        if param_dist is None:
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [0, 0.01, 0.1, 1]
            }
        
        # Scale features
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Create base model for search
        base_model = XGBClassifier(
            device=self.device,
            eval_metric='mlogloss',
            random_state=self.random_state
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Randomized search
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=tscv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X_scaled, y)
        
        print(f"\n✅ Best CV score: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        # Update model with best estimator
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.is_fitted = True
        
        return self.best_params
    
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() or tune() first.")
        
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() or tune() first.")
        
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 scale: bool = True) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Returns:
            Dict with accuracy, f1_score, classification_report, confusion_matrix
        """
        y_pred = self.predict(X, scale=scale)
        
        labels = ['DOWN', 'SIDEWAYS', 'UP'] if self.n_classes == 3 else ['DOWN', 'UP']
        
        self.metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'classification_report': classification_report(y, y_pred, 
                                                          target_names=labels,
                                                          output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            names = self.feature_names
        else:
            names = [f'feature_{i}' for i in range(len(importance))]
        
        fi_df = pd.DataFrame({
            'Feature': names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return fi_df
    
    def save(self, model_dir: str, name: str = 'xgb_baseline'):
        """
        Save model and all artifacts.
        
        Saves:
        - model.joblib: XGBoost model
        - scaler.joblib: Feature scaler
        - feature_names.json: List of feature names
        - metrics.json: Evaluation metrics
        - best_params.json: Best hyperparameters
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_dir / f'{name}_model.joblib')
        
        # Save scaler
        joblib.dump(self.scaler, model_dir / f'{name}_scaler.joblib')
        
        # Save feature names
        with open(model_dir / f'{name}_features.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save metrics
        if self.metrics:
            with open(model_dir / f'{name}_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # Save best params
        if self.best_params:
            with open(model_dir / f'{name}_params.json', 'w') as f:
                json.dump(self.best_params, f, indent=2)
        
        print(f"✅ Model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str, name: str = 'xgb_baseline') -> 'XGBBaseline':
        """Load model from directory."""
        model_dir = Path(model_dir)
        
        instance = cls()
        
        # Load model
        instance.model = joblib.load(model_dir / f'{name}_model.joblib')
        
        # Load scaler
        instance.scaler = joblib.load(model_dir / f'{name}_scaler.joblib')
        
        # Load feature names
        features_path = model_dir / f'{name}_features.json'
        if features_path.exists():
            with open(features_path, 'r') as f:
                instance.feature_names = json.load(f)
        
        # Load metrics
        metrics_path = model_dir / f'{name}_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                instance.metrics = json.load(f)
        
        # Load best params
        params_path = model_dir / f'{name}_params.json'
        if params_path.exists():
            with open(params_path, 'r') as f:
                instance.best_params = json.load(f)
        
        instance.is_fitted = True
        
        print(f"✅ Model loaded from {model_dir}")
        
        return instance


def print_classification_report(metrics: Dict[str, Any], title: str = "Evaluation Results"):
    """Pretty print classification metrics."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print('='*60)
    
    print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"F1 Macro:    {metrics['f1_macro']:.4f}")
    
    print("\nClassification Report:")
    report = metrics['classification_report']
    
    # Print per-class metrics
    for label in ['DOWN', 'SIDEWAYS', 'UP']:
        if label.lower() in report:
            r = report[label.lower()]
        elif label in report:
            r = report[label]
        else:
            continue
        print(f"  {label:10s}: precision={r['precision']:.3f}, "
              f"recall={r['recall']:.3f}, f1={r['f1-score']:.3f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    model = XGBBaseline(device='cpu')  # Use CPU for test
    model.fit(X_train, y_train, feature_names=[f'f{i}' for i in range(n_features)])
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print_classification_report(metrics)
    
    # Feature importance
    fi = model.get_feature_importance()
    print("\nTop 10 Features:")
    print(fi.head(10))
