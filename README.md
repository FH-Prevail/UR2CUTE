# UR2CUTE

**Using Repetitively 2 CNNs for Unsteady Timeseries Estimation**

UR2CUTE is a specialized forecasting model designed for intermittent time series data. By employing a dual CNN approach with PyTorch, it effectively addresses the challenges of predicting both the occurrence and magnitude of demand in irregular time series patterns.

## 📋 Overview

Intermittent demand forecasting presents unique challenges due to irregular and unpredictable demand patterns, characterized by periods of zero demand followed by random non-zero demand. Traditional forecasting methods often perform poorly on such data.

UR2CUTE employs a two-step approach:
1. A CNN-based classification model predicts demand occurrence (zero vs. non-zero)
2. A CNN-based regression model estimates the magnitude of demand

This dual-phase approach significantly improves forecasting accuracy for intermittent demand, particularly in predicting periods of zero demand.

## 🔍 Features

- **PyTorch Implementation**: Uses PyTorch's efficient tensor operations and GPU acceleration
- **Two-Step Prediction Process**: Separate models for order occurrence and quantity prediction
- **Temporal Pattern Recognition**: CNNs effectively capture temporal patterns in intermittent data
- **Lag Feature Generation**: Automatically creates lagged features to capture historical dependencies
- **Combined Loss Function**: Custom loss functions optimized for each prediction task
- **Sklearn Compatibility**: Follows scikit-learn API conventions for easy integration
- **Direct Multi-Step Forecasting**: Predicts multiple future time steps in one pass

## 📦 Dependencies

- Python 3.7+
- PyTorch 1.7+
- NumPy
- pandas
- scikit-learn

## 🛠️ Installation

```bash
pip install fh_prevail
```

Then import the model:

```python
from fh_prevail.models import UR2CUTE
```

## 📊 Quick Start

```python
import pandas as pd
from fh_prevail.models import UR2CUTE

# Load time series data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=50, freq='W'),
    'target': [0, 5, 0, 0, 12, 0, 0, 0, 7, 0, ...],  # Intermittent data
    'feat1': [...],  # Optional external features
    'feat2': [...]   # Optional external features
})

# Initialize model
model = UR2CUTE(
    n_steps_lag=3,
    forecast_horizon=4,
    external_features=['feat1', 'feat2']
)

# Fit model
model.fit(df, target_col='target')

# Make predictions for the next forecast_horizon steps
predictions = model.predict(df)
print("Predicted values:", predictions)

# Note: The model automatically uses GPU if available
# To manually control device placement:
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
```

## 🔧 Parameters

| Parameter                | Description                                                                                                                                   | Default |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `n_steps_lag`            | Number of lag features to generate                                                                                                            | 3       |
| `forecast_horizon`       | Number of future steps to predict                                                                                                             | 8       |
| `external_features`      | List of column names for external features                                                                                                    | None    |
| `epochs`                 | Training epochs for both CNN models                                                                                                           | 100     |
| `batch_size`             | Batch size for training                                                                                                                       | 32      |
| `threshold`              | Probability threshold for classifying zero vs. non-zero. Can be a float or `"auto"` to automatically compute the threshold from training targets | 0.5     |
| `patience`               | Patience for EarlyStopping                                                                                                                    | 10      |
| `random_seed`            | Random seed for reproducibility                                                                                                               | 42      |
| `classification_lr`      | Learning rate for classification model                                                                                                        | 0.0021  |
| `regression_lr`          | Learning rate for regression model                                                                                                            | 0.0021  |
| `dropout_classification` | Dropout rate for classification model                                                                                                         | 0.4     |
| `dropout_regression`     | Dropout rate for regression model                                                                                                             | 0.2     |


## 📝 Methods

### fit(df, target_col)

Fits the UR2CUTE model on the time series data.

**Parameters:**
- `df` (pandas.DataFrame): Time series data with at least the target column. Must be sorted by time.
- `target_col` (str): The name of the column to forecast.

**Returns:**
- The fitted UR2CUTE model instance.

### predict(df)

Predicts the next `forecast_horizon` steps from the input DataFrame.

**Parameters:**
- `df` (pandas.DataFrame): Time series data with the same columns as used in fit(). Must be sorted by time.

**Returns:**
- numpy.ndarray: The predictions for each step in the horizon.

## 🔍 How It Works

1. **Data Preprocessing**:
   - Aggregates demand data (e.g., daily to weekly)
   - Generates lag features to capture historical patterns

2. **Model Architecture**:
   - **Classification Model**: CNN that predicts probability of non-zero demand
   - **Regression Model**: CNN that predicts quantity when demand exists

3. **Prediction Process**:
   - Classification model predicts if demand will occur
   - Regression model predicts the magnitude of demand
   - Final prediction combines both models' outputs

## 🏆 Performance

UR2CUTE outperforms traditional forecasting techniques including:
- Croston's method
- XGBoost
- Random Forest
- ETR
- Prophet
- AutoARIMA

Particularly for predicting intermittent demand, UR2CUTE shows significant improvements in:
- Mean Absolute Error % (MAE%)
- Root Mean Square Error % (RMSE%)
- R-squared values

## 📚 Citation

If you use UR2CUTE in your research, please cite:

```
@article{mirshahi2024intermittent,
  title={Intermittent Time Series Demand Forecasting Using Dual Convolutional Neural Networks},
  author={Mirshahi, Sina and Brandtner, Patrick and Kom{\'i}nkov{\'a} Oplatkov{\'a}, Zuzana},
  journal={MENDEL — Soft Computing Journal},
  volume={30},
  number={1},
  year={2024},
  publisher={MENDEL Journal}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Sina Mirshahi
- Patrick Brandtner
- Zuzana Komínková Oplatková
- Taha Falatouri
- Mehran Naseri
- Farzaneh Darbanian

## 🙏 Acknowledgments

This research was conducted at:
- Department of Informatics and Artificial Intelligence, Tomas Bata
- Department for Logistics, University of Applied Sciences Upper Austria, Steyr
- Josef Ressel-Centre for Predictive Value Network Intelligence, Steyr