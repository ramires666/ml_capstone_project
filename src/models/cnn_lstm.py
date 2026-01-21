"""
CNN-LSTM Model for Bitcoin Price Prediction

Architecture based on the paper:
"Bitcoin price direction prediction using on-chain data"

Uses PyTorch with CUDA for GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features array of shape (samples, lookback, features)
            y: Labels array of shape (samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNLSTM(nn.Module):
    """
    CNN-LSTM architecture for time series classification.
    
    Architecture:
    1. Conv1D → BatchNorm → ReLU → Dropout
    2. LSTM layer 1 → BatchNorm → Dropout
    3. LSTM layer 2
    4. Dense → ReLU → Dense → Softmax
    """
    
    def __init__(self,
                 n_features: int,
                 n_classes: int = 3,
                 conv_filters: int = 16,
                 kernel_size: int = 3,
                 lstm_units: int = 64,
                 dropout: float = 0.5,
                 dense_units: int = 32):
        """
        Args:
            n_features: Number of input features
            n_classes: Number of output classes (2 or 3)
            conv_filters: Number of convolutional filters
            kernel_size: Convolution kernel size
            lstm_units: Units in first LSTM layer
            dropout: Dropout rate
            dense_units: Units in dense layer
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        
        # CNN block
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM blocks
        self.lstm1 = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=False
        )
        self.bn2 = nn.BatchNorm1d(lstm_units)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_units,
            hidden_size=lstm_units * 2,
            batch_first=True,
            bidirectional=False
        )
        
        # Dense output
        self.fc1 = nn.Linear(lstm_units * 2, dense_units)
        self.fc2 = nn.Linear(dense_units, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, lookback, features)
        
        Returns:
            Output logits of shape (batch, n_classes)
        """
        # x shape: (batch, lookback, features)
        batch_size = x.size(0)
        
        # Transpose for Conv1D: (batch, features, lookback)
        x = x.permute(0, 2, 1)
        
        # CNN block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Back to (batch, lookback, conv_filters)
        x = x.permute(0, 2, 1)
        
        # LSTM 1
        x, _ = self.lstm1(x)
        
        # Take last timestep and apply batch norm
        x_last = x[:, -1, :]  # (batch, lstm_units)
        x_last = self.bn2(x_last)
        x_last = self.dropout2(x_last)
        
        # LSTM 2 (single step)
        x_seq = x_last.unsqueeze(1)  # (batch, 1, lstm_units)
        x, _ = self.lstm2(x_seq)
        x = x[:, -1, :]  # (batch, lstm_units * 2)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class CNNLSTMModel:
    """
    Wrapper class for training and inference with CNN-LSTM.
    
    Handles:
    - Data preprocessing and sequence creation
    - Training with early stopping
    - Evaluation and metrics
    - Save/load functionality
    """
    
    def __init__(self,
                 n_classes: int = 3,
                 lookback: int = 20,
                 conv_filters: int = 16,
                 lstm_units: int = 64,
                 dropout: float = 0.5,
                 dense_units: int = 32,
                 learning_rate: float = 0.001,
                 device: str = 'cuda',
                 random_seed: int = 42):
        """
        Initialize CNN-LSTM model.
        
        Args:
            n_classes: Number of classes (2 or 3)
            lookback: Sequence length for LSTM
            conv_filters: Conv1D filters
            lstm_units: LSTM hidden units
            dropout: Dropout rate
            dense_units: Dense layer units
            learning_rate: Learning rate for Adam
            device: 'cuda' or 'cpu'
            random_seed: Random seed
        """
        self.n_classes = n_classes
        self.lookback = lookback
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        # Set device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU")
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        self.model: Optional[CNNLSTM] = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.is_fitted = False
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM from tabular data."""
        n_samples = len(X) - self.lookback
        n_features = X.shape[1]
        
        X_seq = np.zeros((n_samples, self.lookback, n_features))
        y_seq = np.zeros(n_samples, dtype=np.int64)
        
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
        
        Args:
            X_train: Training features (samples, features)
            y_train: Training labels (samples,)
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            epochs: Max training epochs
            batch_size: Batch size
            patience: Early stopping patience
            scale: Whether to scale features
        
        Returns:
            self
        """
        if feature_names:
            self.feature_names = feature_names
        
        n_features = X_train.shape[1]
        
        # Scale features
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Create sequences
        print(f"\n📊 Creating sequences (lookback={self.lookback})...")
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)
        
        if X_val_scaled is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
        else:
            X_val_seq, y_val_seq = None, None
        
        print(f"   Train sequences: {X_train_seq.shape}")
        if X_val_seq is not None:
            print(f"   Val sequences: {X_val_seq.shape}")
        
        # Create dataloaders
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val_seq is not None:
            val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = CNNLSTM(
            n_features=n_features,
            n_classes=self.n_classes,
            conv_filters=self.conv_filters,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
            dense_units=self.dense_units
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        print(f"\n🏋️ Training for {epochs} epochs (patience={patience})...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        
                        preds = outputs.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(y_batch.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_acc = accuracy_score(all_labels, all_preds)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Print progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}")
        
        # Load best state
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True
        print("✅ Training complete!")
        
        return self
    
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Predict
        self.model.eval()
        dataset = TimeSeriesDataset(X_seq, np.zeros(len(X_seq), dtype=np.int64))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        return np.array(all_preds)
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        self.model.eval()
        dataset = TimeSeriesDataset(X_seq, np.zeros(len(X_seq), dtype=np.int64))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, scale: bool = True) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(X, scale=scale)
        
        # Adjust y to match predictions (due to lookback)
        y_eval = y[self.lookback:]
        
        labels = ['DOWN', 'SIDEWAYS', 'UP'] if self.n_classes == 3 else ['DOWN', 'UP']
        
        self.metrics = {
            'accuracy': accuracy_score(y_eval, y_pred),
            'f1_weighted': f1_score(y_eval, y_pred, average='weighted'),
            'f1_macro': f1_score(y_eval, y_pred, average='macro'),
            'classification_report': classification_report(y_eval, y_pred,
                                                          target_names=labels,
                                                          output_dict=True),
            'confusion_matrix': confusion_matrix(y_eval, y_pred).tolist()
        }
        
        return self.metrics
    
    def save(self, model_dir: str, name: str = 'cnn_lstm'):
        """Save model and all artifacts."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), model_dir / f'{name}_model.pt')
        
        # Save scaler
        joblib.dump(self.scaler, model_dir / f'{name}_scaler.joblib')
        
        # Save config
        config = {
            'n_classes': self.n_classes,
            'lookback': self.lookback,
            'conv_filters': self.conv_filters,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'dense_units': self.dense_units,
            'n_features': self.model.n_features if self.model else None,
            'feature_names': self.feature_names
        }
        with open(model_dir / f'{name}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save metrics
        if self.metrics:
            with open(model_dir / f'{name}_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # Save history
        with open(model_dir / f'{name}_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✅ Model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str, name: str = 'cnn_lstm', device: str = 'cuda') -> 'CNNLSTMModel':
        """Load model from directory."""
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
        instance.feature_names = config.get('feature_names', [])
        
        # Load scaler
        instance.scaler = joblib.load(model_dir / f'{name}_scaler.joblib')
        
        # Load model
        instance.model = CNNLSTM(
            n_features=config['n_features'],
            n_classes=config['n_classes'],
            conv_filters=config['conv_filters'],
            lstm_units=config['lstm_units'],
            dropout=config['dropout'],
            dense_units=config['dense_units']
        ).to(instance.device)
        
        instance.model.load_state_dict(
            torch.load(model_dir / f'{name}_model.pt', map_location=instance.device)
        )
        
        # Load metrics
        metrics_path = model_dir / f'{name}_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                instance.metrics = json.load(f)
        
        instance.is_fitted = True
        
        print(f"✅ Model loaded from {model_dir}")
        
        return instance


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 30
    lookback = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    model = CNNLSTMModel(device='cpu', lookback=lookback)
    model.fit(X_train, y_train, X_test, y_test, epochs=20, patience=5)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1:       {metrics['f1_weighted']:.4f}")
