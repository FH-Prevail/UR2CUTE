import os
import random
import numpy as np
import pandas as pd
import math

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# Utility functions (unchanged)
# -------------------------
def _generate_lag_features(df, column_name, n_lags=1):
    """
    Generate lag features for a given column in the dataframe.
    """
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"{column_name}_Lag{i}"] = df[column_name].shift(i)
    return df

def _create_multistep_data(df, target_name, external_features, n_steps_lag, forecast_horizon):
    """
    Build multi-step training samples.
    For each row i (up to len(df)-forecast_horizon):
      - Input: [external features] + [lag features from row i]
      - Target: next forecast_horizon values of target_name (rows i+1 .. i+forecast_horizon).
    """
    X_list = []
    y_list = []
    for i in range(len(df) - forecast_horizon):
        lag_vals = df.iloc[i][[f"{target_name}_Lag{j}" for j in range(1, n_steps_lag + 1)]].values

        if external_features:
            ext_vals = df.loc[i, external_features].values
        else:
            ext_vals = []
        
        X_list.append(np.concatenate([ext_vals, lag_vals]))
        y_seq = df.loc[i+1 : i+forecast_horizon, target_name].values
        y_list.append(y_seq)
    return np.array(X_list), np.array(y_list)


# -------------------------
# PyTorch model definitions
# -------------------------
class _Classifier(nn.Module):
    def __init__(self, n_features, forecast_horizon, dropout_rate):
        super(_Classifier, self).__init__()
        # Input expected shape: (batch, 1, n_features)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Determine the flattened size
        dummy = torch.zeros(1, 1, n_features)
        out = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(dummy)))))
        flattened_size = out.numel()
        
        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, forecast_horizon)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, 1, n_features)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class _Regressor(nn.Module):
    def __init__(self, n_features, forecast_horizon, dropout_rate):
        super(_Regressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        
        dummy = torch.zeros(1, 1, n_features)
        out = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(dummy)))))
        flattened_size = out.numel()
        
        self.fc1 = nn.Linear(flattened_size, 46)
        self.fc2 = nn.Linear(46, forecast_horizon)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------
# UR2CUTE estimator rewritten with PyTorch
# -------------------------
class UR2CUTE(BaseEstimator):
    """
    UR2CUTE: Using Repetitively 2 CNNs for Unsteady Timeseries Estimation with a two-step/hurdle approach.
    This estimator performs multi-step forecasting with:
      - A CNN-based classification model to predict zero vs. nonzero demand.
      - A CNN-based regression model to predict the quantity (trained only on sequences with any demand).
    """
    def __init__(
        self,
        n_steps_lag=3,
        forecast_horizon=8,
        external_features=None,
        epochs=100,
        batch_size=32,
        threshold=0.5,
        patience=10,
        random_seed=42,
        classification_lr=0.0021,
        regression_lr=0.0021,
        dropout_classification=0.4,
        dropout_regression=0.2
    ):
        self.n_steps_lag = n_steps_lag
        self.forecast_horizon = forecast_horizon
        self.external_features = external_features if external_features is not None else []
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.patience = patience
        self.random_seed = random_seed
        self.classification_lr = classification_lr
        self.regression_lr = regression_lr
        self.dropout_classification = dropout_classification
        self.dropout_regression = dropout_regression

        self.classifier_ = None
        self.regressor_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None
        self.n_features_ = None
        self.target_col_ = None  # set during fit

    def _set_random_seeds(self):
        """
        Force reproducible behavior by setting seeds.
        """
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def _train_model(self, model, optimizer, loss_fn, train_loader, val_loader, device):
        best_val_loss = np.inf
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    outputs = model(X_val)
                    loss = loss_fn(outputs, y_val)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Uncomment the next line to see epoch progress
            # print(f"Epoch {epoch+1}/{self.epochs} - Train loss: {avg_train_loss:.4f} - Val loss: {avg_val_loss:.4f}")
            
            if patience_counter >= self.patience:
                # print("Early stopping triggered.")
                break
        
        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def fit(self, df, target_col):
        """
        Fit the UR2CUTE model on a time-series dataframe.
        """
        self._set_random_seeds()
        self.target_col_ = target_col
        
        # 1) Generate lag features & drop NaNs
        df_lagged = _generate_lag_features(df, target_col, n_lags=self.n_steps_lag)
        df_lagged.dropna(inplace=True)
        df_lagged.reset_index(drop=True, inplace=True)
        
        # 2) Create multi-step training data
        X_all, y_all = _create_multistep_data(
            df_lagged,
            target_col,
            self.external_features,
            self.n_steps_lag,
            self.forecast_horizon
        )
        # 3) Scale inputs
        self.scaler_X_ = MinMaxScaler()
        X_scaled = self.scaler_X_.fit_transform(X_all)
        
        self.scaler_y_ = MinMaxScaler()
        y_flat = y_all.flatten().reshape(-1, 1)
        self.scaler_y_.fit(y_flat)
        y_scaled = self.scaler_y_.transform(y_flat).reshape(y_all.shape)
        
        # For CNN in PyTorch, we want input shape (batch, channels, features)
        # Here channels=1 and features = number of features per sample.
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        self.n_features_ = X_reshaped.shape[2]
        
        # 4) Time-based split for validation (90% train, 10% val)
        val_split_idx = int(len(X_reshaped) * 0.9)
        X_train = X_reshaped[:val_split_idx]
        y_train = y_all[:val_split_idx]
        X_val = X_reshaped[val_split_idx:]
        y_val = y_all[val_split_idx:]
        
        y_train_scaled = y_scaled[:val_split_idx]
        y_val_scaled = y_scaled[val_split_idx:]
        
        # Classification target: binary (zero vs nonzero)
        y_train_binary = (y_train > 0).astype(np.float32)
        y_val_binary = (y_val > 0).astype(np.float32)
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --------------------------
        # Build & train Classification Model
        # --------------------------
        self.classifier_ = _Classifier(n_features=self.n_features_, 
                                       forecast_horizon=self.forecast_horizon, 
                                       dropout_rate=self.dropout_classification)
        self.classifier_.to(device)
        
        # Create DataLoader for classification
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_binary, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_binary, dtype=torch.float32)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        optimizer_cls = optim.Adam(self.classifier_.parameters(), lr=self.classification_lr)
        loss_fn_cls = nn.BCELoss()
        
        self.classifier_ = self._train_model(self.classifier_, optimizer_cls, loss_fn_cls, train_loader, val_loader, device)
        
        # --------------------------
        # Build & train Regression Model
        # Train only on samples that have at least one nonzero step in the horizon.
        # --------------------------
        # Filter training samples with sum > 0
        nonzero_mask_train = (y_train.sum(axis=1) > 0)
        nonzero_mask_val = (y_val.sum(axis=1) > 0)
        
        X_train_reg = X_train[nonzero_mask_train]
        y_train_reg = y_train_scaled[nonzero_mask_train]
        
        X_val_reg = X_val[nonzero_mask_val]
        y_val_reg = y_val_scaled[nonzero_mask_val]
        
        self.regressor_ = _Regressor(n_features=self.n_features_, 
                                     forecast_horizon=self.forecast_horizon, 
                                     dropout_rate=self.dropout_regression)
        self.regressor_.to(device)
        
        # Create DataLoader for regression
        X_train_reg_tensor = torch.tensor(X_train_reg, dtype=torch.float32)
        y_train_reg_tensor = torch.tensor(y_train_reg, dtype=torch.float32)
        train_reg_dataset = TensorDataset(X_train_reg_tensor, y_train_reg_tensor)
        train_reg_loader = DataLoader(train_reg_dataset, batch_size=self.batch_size, shuffle=True)
        
        X_val_reg_tensor = torch.tensor(X_val_reg, dtype=torch.float32)
        y_val_reg_tensor = torch.tensor(y_val_reg, dtype=torch.float32)
        val_reg_dataset = TensorDataset(X_val_reg_tensor, y_val_reg_tensor)
        val_reg_loader = DataLoader(val_reg_dataset, batch_size=self.batch_size, shuffle=False)
        
        optimizer_reg = optim.Adam(self.regressor_.parameters(), lr=self.regression_lr)
        loss_fn_reg = nn.MSELoss()
        
        self.regressor_ = self._train_model(self.regressor_, optimizer_reg, loss_fn_reg, train_reg_loader, val_reg_loader, device)
        
        return self

    def predict(self, df):
        """
        Predict the next forecast_horizon steps from the *last* row of the input DataFrame.
        """
        # Assume the target column is the one used in fit.
        target_col = self.target_col_ if self.target_col_ is not None else "target"
        df_lagged = _generate_lag_features(df, target_col, n_lags=self.n_steps_lag)
        df_lagged.dropna(inplace=True)
        
        last_idx = df_lagged.index[-1]
        lag_vals = df_lagged.loc[last_idx, [f"{target_col}_Lag{j}" for j in range(1, self.n_steps_lag + 1)]].values
        
        if self.external_features:
            ext_vals = df_lagged.loc[last_idx, self.external_features].values
        else:
            ext_vals = []
        
        x_input = np.concatenate([ext_vals, lag_vals]).reshape(1, -1)
        x_input_scaled = self.scaler_X_.transform(x_input)
        # Reshape for PyTorch: (batch, channels, features)
        x_input_reshaped = x_input_scaled.reshape((1, 1, x_input_scaled.shape[1]))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = torch.tensor(x_input_reshaped, dtype=torch.float32).to(device)
        
        # Classification prediction
        self.classifier_.eval()
        with torch.no_grad():
            order_prob = self.classifier_(x_tensor).cpu().numpy()[0]  # shape: (forecast_horizon,)
        
        # Regression prediction
        self.regressor_.eval()
        with torch.no_grad():
            quantity_pred_scaled = self.regressor_(x_tensor).cpu().numpy()[0]  # shape: (forecast_horizon,)
        
        # Inverse scale regression prediction
        quantity_pred = self.scaler_y_.inverse_transform(quantity_pred_scaled.reshape(-1, 1)).flatten()
        
        # Combine based on threshold
        final_preds = []
        for prob, qty in zip(order_prob, quantity_pred):
            pred = qty if prob > self.threshold else 0
            final_preds.append(max(0, round(pred)))
        
        return np.array(final_preds)

    def get_params(self, deep=True):
        return {
            'n_steps_lag': self.n_steps_lag,
            'forecast_horizon': self.forecast_horizon,
            'external_features': self.external_features,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'threshold': self.threshold,
            'patience': self.patience,
            'random_seed': self.random_seed,
            'classification_lr': self.classification_lr,
            'regression_lr': self.regression_lr,
            'dropout_classification': self.dropout_classification,
            'dropout_regression': self.dropout_regression
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ---------------
# Example usage:
# ---------------
if __name__ == "__main__":
    # Create a simple synthetic example.
    df_example = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=50, freq='W'),
        'target': np.random.randint(0, 20, 50),
        'feat1': np.random.randn(50) * 10,
        'feat2': np.random.randn(50) * 5
    }).sort_values('date').reset_index(drop=True)

    # Initialize UR2CUTE.
    model = UR2CUTE(
        n_steps_lag=3,
        forecast_horizon=4,
        external_features=['feat1', 'feat2'],
        epochs=5,  # use fewer epochs for demonstration
        batch_size=8,
        threshold=0.6
    )
    # Fit on the entire data (using column "target")
    model.fit(df_example, target_col='target')

    # Predict the next 4 steps from the final row
    preds = model.predict(df_example)
    print("Predicted horizon:", preds)
