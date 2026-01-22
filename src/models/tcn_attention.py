"""
TCN-Attention Model for Bitcoin Price Direction Prediction.

This module implements a Temporal Convolutional Network (TCN) with Multi-Head
Self-Attention, designed as an alternative to CNN-LSTM.

Why TCN instead of LSTM:
- Parallelizable (faster training)
- No vanishing gradient problem
- Explicit control over receptive field via dilation
- Often better on financial time series

Why Attention:
- Learns which time steps are most important
- Captures long-range dependencies better than convolutions alone
- Interpretable attention weights

Usage:
    model = TCNAttentionModel(lookback=32, device='cuda')
    model.fit(X_train, y_train, X_val, y_val, epochs=50)
    metrics = model.evaluate(X_test, y_test)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def check_gpu():
    """Check GPU availability."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU detected: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    else:
        print("💻 No GPU detected, using CPU")
        return False


class TCNBlock(layers.Layer):
    """
    Temporal Convolutional Block with dilated causal convolution.
    
    Dilation allows the network to have a large receptive field with
    fewer layers. For example with kernel_size=3:
    - Dilation 1: sees 3 steps
    - Dilation 2: sees 5 steps  
    - Dilation 4: sees 9 steps
    """
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout
        
        # Dilated causal convolution
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='causal',  # Only look at past/current, never future
            dilation_rate=dilation_rate,
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.dropout = layers.Dropout(dropout)
        
        # 1x1 conv for residual connection if dimensions don't match
        self.residual_conv = None
    
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.residual_conv = layers.Conv1D(self.filters, 1, padding='same')
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        # Residual connection
        if self.residual_conv is not None:
            res = self.residual_conv(inputs)
        else:
            res = inputs
        
        return x + res


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention layer.
    
    Allows the model to focus on different parts of the sequence
    when making predictions. Multiple heads let it attend to
    different types of patterns simultaneously.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.layernorm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)
    
    def call(self, inputs, training=None):
        # Self-attention: query, key, value are all the same input
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout(attn_output, training=training)
        # Residual connection + layer norm
        return self.layernorm(inputs + attn_output)


def build_tcn_attention_model(
    n_features: int,
    n_classes: int = 3,
    lookback: int = 32,
    tcn_filters: int = 64,
    kernel_size: int = 3,
    num_tcn_blocks: int = 3,
    attention_heads: int = 4,
    dropout: float = 0.2,
    dense_units: int = 32
) -> keras.Model:
    """
    Build TCN-Attention model.
    
    Architecture:
    1. Multiple TCN blocks with increasing dilation
    2. Multi-head self-attention
    3. Global average pooling
    4. Dense + softmax output
    """
    inputs = layers.Input(shape=(lookback, n_features), name='input')
    
    x = inputs
    
    # TCN blocks with exponentially increasing dilation
    # This gives us a large receptive field efficiently
    for i in range(num_tcn_blocks):
        dilation = 2 ** i  # 1, 2, 4, 8, ...
        x = TCNBlock(
            filters=tcn_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout=dropout,
            name=f'tcn_block_{i+1}'
        )(x)
    
    # Multi-head self-attention
    x = MultiHeadSelfAttention(
        embed_dim=tcn_filters,
        num_heads=attention_heads,
        dropout=dropout,
        name='attention'
    )(x)
    
    # Global average pooling to get fixed-size representation
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Dense layer
    x = layers.Dense(dense_units, activation='relu', name='dense')(x)
    x = layers.Dropout(dropout)(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='tcn_attention')
    
    return model


class TCNAttentionModel:
    """
    Wrapper class for TCN-Attention model training and inference.
    
    Features:
    - Class weights for imbalanced data
    - F1-based evaluation
    - Early stopping on validation loss
    - Save/load functionality
    """
    
    def __init__(
        self,
        n_classes: int = 3,
        lookback: int = 32,
        tcn_filters: int = 64,
        kernel_size: int = 3,
        num_tcn_blocks: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.2,
        dense_units: int = 32,
        learning_rate: float = 0.001,
        device: str = 'cuda',
        random_seed: int = 42
    ):
        """Initialize TCN-Attention model."""
        self.n_classes = n_classes
        self.lookback = lookback
        self.tcn_filters = tcn_filters
        self.kernel_size = kernel_size
        self.num_tcn_blocks = num_tcn_blocks
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        if device == 'cuda':
            check_gpu()
        
        self.scaler = RobustScaler()
        self.model: Optional[keras.Model] = None
        self.n_features: int = 0
        self.history: Dict[str, List[float]] = {}
        self.metrics: Dict[str, Any] = {}
        self.is_fitted: bool = False
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create 3D sequences for TCN input."""
        n_samples = len(X) - self.lookback
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, self.lookback, n_features), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.lookback]
            y_seq[i] = y[i + self.lookback]
        
        return X_seq, y_seq
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 128,
        patience: int = 5,
        use_class_weights: bool = True,
        scale: bool = True
    ) -> 'TCNAttentionModel':
        """
        Train the model.
        
        Args:
            use_class_weights: If True, compute class weights to handle imbalance
        """
        self.n_features = X_train.shape[1]
        
        print("\n📊 Preparing data...")
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
            X_val_scaled = self.scaler.transform(X_val).astype(np.float32) if X_val is not None else None
        else:
            X_train_scaled = X_train.astype(np.float32)
            X_val_scaled = X_val.astype(np.float32) if X_val is not None else None
        
        print(f"📊 Creating sequences (lookback={self.lookback})...")
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)
        print(f"   Train sequences: {X_train_seq.shape}")
        
        if X_val_scaled is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
            print(f"   Val sequences: {X_val_seq.shape}")
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
            y_val_seq = None
        
        # Compute class weights if requested
        class_weight_dict = None
        if use_class_weights:
            classes = np.unique(y_train_seq)
            weights = compute_class_weight('balanced', classes=classes, y=y_train_seq)
            class_weight_dict = dict(zip(classes, weights))
            print(f"   Class weights: {class_weight_dict}")
        
        # Build model
        print("\n🏗️ Building TCN-Attention model...")
        self.model = build_tcn_attention_model(
            n_features=self.n_features,
            n_classes=self.n_classes,
            lookback=self.lookback,
            tcn_filters=self.tcn_filters,
            kernel_size=self.kernel_size,
            num_tcn_blocks=self.num_tcn_blocks,
            attention_heads=self.attention_heads,
            dropout=self.dropout,
            dense_units=self.dense_units
        )
        
        # Compile with F1-friendly settings
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        print(f"\n🏋️ Training for up to {epochs} epochs (patience={patience})...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
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
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if scale:
            X_scaled = self.scaler.transform(X).astype(np.float32)
        else:
            X_scaled = X.astype(np.float32)
        
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        proba = self.model.predict(X_seq, verbose=0)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if scale:
            X_scaled = self.scaler.transform(X).astype(np.float32)
        else:
            X_scaled = X.astype(np.float32)
        
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        return self.model.predict(X_seq, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, scale: bool = True) -> Dict[str, Any]:
        """Evaluate model, focusing on F1 scores."""
        y_pred = self.predict(X, scale=scale)
        y_eval = y[self.lookback:]
        
        labels = ['DOWN', 'SIDEWAYS', 'UP'] if self.n_classes == 3 else ['DOWN', 'UP']
        
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
    
    def save(self, model_dir: str, name: str = 'tcn_attention') -> None:
        """Save model and artifacts."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save(model_dir / f'{name}_model.keras')
        joblib.dump(self.scaler, model_dir / f'{name}_scaler.joblib')
        
        config = {
            'n_classes': self.n_classes,
            'lookback': self.lookback,
            'tcn_filters': self.tcn_filters,
            'kernel_size': self.kernel_size,
            'num_tcn_blocks': self.num_tcn_blocks,
            'attention_heads': self.attention_heads,
            'dropout': self.dropout,
            'dense_units': self.dense_units,
            'n_features': self.n_features
        }
        with open(model_dir / f'{name}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.metrics:
            with open(model_dir / f'{name}_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str, name: str = 'tcn_attention', 
             device: str = 'cuda') -> 'TCNAttentionModel':
        """Load model from disk."""
        model_dir = Path(model_dir)
        
        with open(model_dir / f'{name}_config.json', 'r') as f:
            config = json.load(f)
        
        instance = cls(
            n_classes=config['n_classes'],
            lookback=config['lookback'],
            tcn_filters=config['tcn_filters'],
            kernel_size=config['kernel_size'],
            num_tcn_blocks=config['num_tcn_blocks'],
            attention_heads=config['attention_heads'],
            dropout=config['dropout'],
            dense_units=config['dense_units'],
            device=device
        )
        instance.n_features = config['n_features']
        
        instance.scaler = joblib.load(model_dir / f'{name}_scaler.joblib')
        instance.model = keras.models.load_model(
            model_dir / f'{name}_model.keras',
            custom_objects={
                'TCNBlock': TCNBlock,
                'MultiHeadSelfAttention': MultiHeadSelfAttention
            }
        )
        
        metrics_path = model_dir / f'{name}_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                instance.metrics = json.load(f)
        
        instance.is_fitted = True
        print(f"✅ Model loaded from {model_dir}")
        
        return instance


if __name__ == "__main__":
    """Quick test with synthetic data."""
    print("=" * 60)
    print("Testing TCN-Attention Module")
    print("=" * 60)
    
    check_gpu()
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    lookback = 20
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("\nTraining model...")
    model = TCNAttentionModel(
        n_classes=3,
        lookback=lookback,
        tcn_filters=32,
        device='cpu'
    )
    
    model.fit(X_train, y_train, X_test, y_test, epochs=5, patience=3)
    
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\n✅ TCN-Attention module test complete!")
