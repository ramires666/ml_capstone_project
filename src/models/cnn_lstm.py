"""
CNN-LSTM Model for Bitcoin Price Direction Prediction.

This module implements a hybrid CNN-LSTM neural network using TensorFlow/Keras.
The architecture combines:
- Conv1D: Extracts local patterns and features from time series
- LSTM: Captures long-term temporal dependencies

Architecture Overview:
----------------------
Input (batch, lookback, features)
    ↓
Conv1D (filters, kernel_size=3)
    ↓
BatchNormalization + ReLU + Dropout
    ↓
LSTM Layer 1 (return_sequences=True)
    ↓
BatchNormalization + Dropout
    ↓
LSTM Layer 2 (return final state only)
    ↓
Dense (hidden units) + ReLU
    ↓
Dense (n_classes) + Softmax
    ↓
Output (batch, n_classes)

Why CNN-LSTM?
-------------
- CNN captures local patterns (like candlestick patterns, short-term momentum)
- LSTM captures long-term dependencies (trends, regime changes)
- Combination often outperforms either alone on financial data

Usage Example:
-------------
    from src.models.cnn_lstm import CNNLSTMModel
    
    model = CNNLSTMModel(lookback=20, device='cuda')
    model.fit(X_train, y_train, X_val, y_val, epochs=100)
    predictions = model.predict(X_test)

Author: Capstone Project
Date: 2026
"""

import os
# Suppress TensorFlow logging (set before importing TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def check_gpu():
    """
    Check if GPU is available for TensorFlow.
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU detected: {gpus[0].name}")
        # Enable memory growth to avoid allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    else:
        print("💻 No GPU detected, using CPU")
        return False


def build_cnn_lstm_model(n_features: int,
                         n_classes: int = 3,
                         lookback: int = 20,
                         conv_filters: int = 32,  # Increased from 16
                         kernel_size: int = 3,
                         lstm_units: int = 64,
                         dropout: float = 0.2,  # Reduced from 0.5 per expert recommendation
                         dense_units: int = 32) -> keras.Model:
    """
    Build CNN-LSTM model using Keras Functional API.
    
    This function creates the neural network architecture.
    The model is NOT trained here - just the architecture is defined.
    
    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    n_classes : int, default=3
        Number of output classes (2 for UP/DOWN, 3 for UP/SIDEWAYS/DOWN).
    lookback : int, default=20
        Number of timesteps in input sequences.
    conv_filters : int, default=32
        Number of filters in first Conv1D layer.
    kernel_size : int, default=3
        Size of 1D convolution kernel.
    lstm_units : int, default=64
        Number of units in first LSTM layer.
    dropout : float, default=0.2
        Dropout rate for regularization (0.2-0.3 recommended for financial data).
    dense_units : int, default=32
        Number of units in dense layer before output.
    
    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    # Input layer: sequences of shape (lookback, n_features)
    inputs = layers.Input(shape=(lookback, n_features), name='input')
    
    # ---------------------------------------------------------------------
    # CNN Block 1: Extract local patterns
    # ---------------------------------------------------------------------
    # Conv1D applies 1D convolution along the time axis
    # This helps detect local patterns like momentum spikes, candle patterns
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding='same',  # Keep sequence length unchanged
        name='conv1d_1'
    )(inputs)
    
    # Batch normalization: stabilize and speed up training
    x = layers.BatchNormalization(name='bn_conv1')(x)
    
    # ReLU activation: introduce non-linearity
    x = layers.Activation('relu', name='relu_conv1')(x)
    
    # Dropout: randomly zero out neurons to prevent overfitting
    x = layers.Dropout(dropout, name='dropout_conv1')(x)
    
    # ---------------------------------------------------------------------
    # CNN Block 2: Deeper feature extraction
    # ---------------------------------------------------------------------
    x = layers.Conv1D(
        filters=conv_filters * 2,  # 64 filters
        kernel_size=kernel_size,
        padding='same',
        name='conv1d_2'
    )(x)
    
    x = layers.BatchNormalization(name='bn_conv2')(x)
    x = layers.Activation('relu', name='relu_conv2')(x)
    x = layers.Dropout(dropout, name='dropout_conv2')(x)
    
    # ---------------------------------------------------------------------
    # LSTM Block 1: Capture temporal dependencies
    # ---------------------------------------------------------------------
    # return_sequences=True means output at every timestep (for stacking LSTMs)
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=True,  # Output sequence for next LSTM
        name='lstm_1'
    )(x)
    
    # Batch normalization after LSTM
    x = layers.BatchNormalization(name='bn_lstm1')(x)
    
    # Dropout
    x = layers.Dropout(dropout, name='dropout_lstm1')(x)
    
    # ---------------------------------------------------------------------
    # LSTM Block 2: Further temporal processing
    # ---------------------------------------------------------------------
    # return_sequences=False means only output the final timestep
    x = layers.LSTM(
        units=lstm_units * 2,  # Double the units for more capacity
        return_sequences=False,  # Only final output
        name='lstm_2'
    )(x)
    
    # ---------------------------------------------------------------------
    # Dense Output Block
    # ---------------------------------------------------------------------
    # Dense layer with ReLU for learning complex combinations
    x = layers.Dense(dense_units, activation='relu', name='dense')(x)
    
    # Output layer: softmax for multi-class classification
    # Outputs probability distribution over classes
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_lstm')
    
    # Compile model with optimizer and loss function
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # For integer labels
        metrics=['accuracy']
    )
    
    return model


class CNNLSTMModel:
    """
    Wrapper class for training and inference with CNN-LSTM.
    
    This class handles:
    - Feature scaling
    - Sequence creation for LSTM input
    - Model training with early stopping
    - Evaluation and metrics calculation
    - Model save/load functionality
    
    The class provides a scikit-learn-like interface (fit, predict, etc.)
    while using TensorFlow/Keras under the hood.
    
    Attributes
    ----------
    model : keras.Model
        The underlying Keras model.
    scaler : RobustScaler
        Scaler for normalizing features.
    history : dict
        Training history (loss, accuracy per epoch).
    metrics : dict
        Evaluation metrics from last evaluate() call.
    
    Example
    -------
    >>> model = CNNLSTMModel(lookback=20, device='cuda')
    >>> model.fit(X_train, y_train, X_val, y_val, epochs=100)
    >>> metrics = model.evaluate(X_test, y_test)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    
    def __init__(self,
                 n_classes: int = 3,
                 lookback: int = 20,
                 conv_filters: int = 32,  # Increased from 16
                 lstm_units: int = 64,
                 dropout: float = 0.2,  # Reduced from 0.5
                 dense_units: int = 32,
                 learning_rate: float = 0.001,
                 device: str = 'cuda',
                 random_seed: int = 42):
        """
        Initialize CNN-LSTM model.
        
        Parameters
        ----------
        n_classes : int, default=3
            Number of output classes.
        lookback : int, default=20
            Sequence length for LSTM input.
        conv_filters : int, default=32
            Number of Conv1D filters.
        lstm_units : int, default=64
            Units in first LSTM layer.
        dropout : float, default=0.2
            Dropout rate (0.2-0.3 recommended for financial data).
        dense_units : int, default=32
            Units in dense layer.
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer.
        device : str, default='cuda'
            'cuda' for GPU, 'cpu' for CPU only.
        random_seed : int, default=42
            Random seed for reproducibility.
        """
        # Store configuration
        self.n_classes = n_classes
        self.lookback = lookback
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Check GPU availability
        if device == 'cuda':
            check_gpu()
        
        # Initialize scaler and metadata
        self.scaler = RobustScaler()
        self.model: Optional[keras.Model] = None
        self.feature_names: List[str] = []
        self.n_features: int = 0
        self.history: Dict[str, List[float]] = {}
        self.metrics: Dict[str, Any] = {}
        self.is_fitted: bool = False
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D sequences for LSTM from 2D tabular data.
        
        Converts (n_samples, n_features) to (n_sequences, lookback, n_features).
        Each sequence contains 'lookback' consecutive timesteps.
        
        Parameters
        ----------
        X : np.ndarray
            2D feature matrix (n_samples, n_features).
        y : np.ndarray
            1D labels (n_samples,).
        
        Returns
        -------
        X_seq : np.ndarray
            3D sequences (n_sequences, lookback, n_features).
        y_seq : np.ndarray
            1D labels for sequences (n_sequences,).
        """
        n_samples = len(X) - self.lookback
        n_features = X.shape[1]
        
        # Pre-allocate arrays
        X_seq = np.zeros((n_samples, self.lookback, n_features), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int32)
        
        # Create sequences using sliding window
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.lookback]
            y_seq[i] = y[i + self.lookback]
        
        return X_seq, y_seq
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None,
            epochs: int = 100,
            batch_size: int = 64,
            patience: int = 15,
            scale: bool = True) -> 'CNNLSTMModel':
        """
        Train the CNN-LSTM model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features (n_samples, n_features).
        y_train : np.ndarray
            Training labels (n_samples,).
        X_val : np.ndarray, optional
            Validation features.
        y_val : np.ndarray, optional
            Validation labels.
        feature_names : List[str], optional
            Names of features for reference.
        epochs : int, default=100
            Maximum training epochs.
        batch_size : int, default=64
            Batch size for training.
        patience : int, default=15
            Early stopping patience (epochs without improvement).
        scale : bool, default=True
            Whether to scale features.
        
        Returns
        -------
        self
        """
        # Store feature info
        if feature_names:
            self.feature_names = feature_names
        self.n_features = X_train.shape[1]
        
        # Scale features
        print("\n📊 Preparing data...")
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
            X_val_scaled = self.scaler.transform(X_val).astype(np.float32) if X_val is not None else None
        else:
            X_train_scaled = X_train.astype(np.float32)
            X_val_scaled = X_val.astype(np.float32) if X_val is not None else None
        
        # Create sequences
        print(f"📊 Creating sequences (lookback={self.lookback})...")
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)
        print(f"   Train sequences: {X_train_seq.shape}")
        
        if X_val_scaled is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
            print(f"   Val sequences: {X_val_seq.shape}")
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Build model
        print("\n🏗️ Building model...")
        self.model = build_cnn_lstm_model(
            n_features=self.n_features,
            n_classes=self.n_classes,
            lookback=self.lookback,
            conv_filters=self.conv_filters,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
            dense_units=self.dense_units
        )
        
        # Update learning rate
        self.model.optimizer.learning_rate.assign(self.learning_rate)
        
        # Print model summary
        self.model.summary()
        
        # Setup callbacks
        callback_list = [
            # Early stopping: stop if val_loss doesn't improve
            # min_delta=0.001 means ignore improvements smaller than 0.1%
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=0.001,  # Stop if improvement < 0.1%
                restore_best_weights=True,  # Keep best model
                verbose=1
            ),
            # Reduce LR on plateau: reduce learning rate if stuck
            # factor=0.3 means LR drops to 30% (more aggressive than 50%)
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # More aggressive reduction (was 0.5)
                patience=5,  # Reduce LR after 5 epochs without improvement
                min_lr=1e-6,
                min_delta=0.001,  # Ignore tiny improvements
                verbose=1
            )
        ]
        
        # Train model
        print(f"\n🏋️ Training for up to {epochs} epochs (patience={patience})...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        # Store training history
        self.history = {
            'train_loss': history.history['loss'],
            'train_acc': history.history['accuracy'],
            'val_loss': history.history.get('val_loss', []),
            'val_acc': history.history.get('val_accuracy', [])
        }
        
        self.is_fitted = True
        print("\n✅ Training complete!")
        
        return self
    
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        scale : bool, default=True
            Whether to scale features.
        
        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale features
        if scale:
            X_scaled = self.scaler.transform(X).astype(np.float32)
        else:
            X_scaled = X.astype(np.float32)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Predict probabilities and take argmax
        proba = self.model.predict(X_seq, verbose=0)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        scale : bool, default=True
            Whether to scale features.
        
        Returns
        -------
        np.ndarray
            Class probabilities (n_sequences, n_classes).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if scale:
            X_scaled = self.scaler.transform(X).astype(np.float32)
        else:
            X_scaled = X.astype(np.float32)
        
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        return self.model.predict(X_seq, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, scale: bool = True) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Parameters
        ----------
        X : np.ndarray
            Test features.
        y : np.ndarray
            Test labels.
        scale : bool, default=True
            Whether to scale features.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with accuracy, F1 scores, confusion matrix, etc.
        """
        # Get predictions
        y_pred = self.predict(X, scale=scale)
        
        # Adjust y to match predictions (due to lookback window)
        y_eval = y[self.lookback:]
        
        # Determine label names
        labels = ['DOWN', 'SIDEWAYS', 'UP'] if self.n_classes == 3 else ['DOWN', 'UP']
        
        # Calculate metrics
        self.metrics = {
            'accuracy': float(accuracy_score(y_eval, y_pred)),
            'f1_weighted': float(f1_score(y_eval, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_eval, y_pred, average='macro')),
            'classification_report': classification_report(
                y_eval, y_pred, target_names=labels, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_eval, y_pred).tolist()
        }
        
        return self.metrics
    
    def save(self, model_dir: str, name: str = 'cnn_lstm') -> None:
        """
        Save model and all artifacts to disk.
        
        Saves:
        - Keras model (SavedModel format)
        - Scaler (joblib)
        - Config (JSON)
        - Metrics (JSON)
        - Training history (JSON)
        
        Parameters
        ----------
        model_dir : str
            Directory to save model.
        name : str, default='cnn_lstm'
            Base name for saved files.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(model_dir / f'{name}_model.keras')
        
        # Save scaler
        joblib.dump(self.scaler, model_dir / f'{name}_scaler.joblib')
        
        # Save configuration
        config = {
            'n_classes': self.n_classes,
            'lookback': self.lookback,
            'conv_filters': self.conv_filters,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'dense_units': self.dense_units,
            'n_features': self.n_features,
            'feature_names': self.feature_names
        }
        with open(model_dir / f'{name}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save metrics
        if self.metrics:
            with open(model_dir / f'{name}_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # Save history
        if self.history:
            with open(model_dir / f'{name}_history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print(f"✅ Model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str, name: str = 'cnn_lstm', 
             device: str = 'cuda') -> 'CNNLSTMModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        model_dir : str
            Directory containing saved model.
        name : str, default='cnn_lstm'
            Base name of saved files.
        device : str, default='cuda'
            Device to use for inference.
        
        Returns
        -------
        CNNLSTMModel
            Loaded model instance.
        """
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / f'{name}_config.json', 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            n_classes=config['n_classes'],
            lookback=config['lookback'],
            conv_filters=config['conv_filters'],
            lstm_units=config['lstm_units'],
            dropout=config['dropout'],
            dense_units=config['dense_units'],
            device=device
        )
        instance.n_features = config['n_features']
        instance.feature_names = config.get('feature_names', [])
        
        # Load scaler
        instance.scaler = joblib.load(model_dir / f'{name}_scaler.joblib')
        
        # Load Keras model
        instance.model = keras.models.load_model(model_dir / f'{name}_model.keras')
        
        # Load metrics if available
        metrics_path = model_dir / f'{name}_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                instance.metrics = json.load(f)
        
        instance.is_fitted = True
        print(f"✅ Model loaded from {model_dir}")
        
        return instance


# =============================================================================
# Module Test
# =============================================================================
if __name__ == "__main__":
    """Quick test with synthetic data."""
    
    print("=" * 60)
    print("Testing CNN-LSTM Module (Keras/TensorFlow)")
    print("=" * 60)
    
    # Check GPU
    check_gpu()
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    lookback = 20
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    print("\nTraining model...")
    model = CNNLSTMModel(
        n_classes=3,
        lookback=lookback,
        lstm_units=32,  # Smaller for quick test
        epochs=10,
        device='cpu'  # Use CPU for test
    )
    
    model.fit(X_train, y_train, X_test, y_test, epochs=10, patience=5)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\n✅ CNN-LSTM module test complete!")
