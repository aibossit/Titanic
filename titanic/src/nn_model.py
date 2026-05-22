import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

RANDOM_STATE = 42


class TitanicNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=[128, 64, 32], dropout=0.3, learning_rate=1e-3,
                 batch_size=32, epochs=300, patience=20, device='cpu', random_state=RANDOM_STATE):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = X.astype(np.float32)
        y = y.astype(np.float32).ravel()
        
        self.classes_ = np.unique(y)
        
        self.input_dim_ = X.shape[1]
        self.model_ = TitanicNN(self.input_dim_, self.hidden_dims, self.dropout).to(self.device)
        
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        criterion = nn.BCELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = self.model_(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                break
        
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
        
        return self

    def predict_proba(self, X):

        self.model_.eval()
        
        if hasattr(X, 'values'):
            X = X.values
        X = X.astype(np.float32)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            proba_positive = self.model_(X_tensor).cpu().numpy()
        
        return np.column_stack([1 - proba_positive, proba_positive])

    def predict(self, X):

        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "device": self.device,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self