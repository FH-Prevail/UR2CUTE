import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

# Function to generate lag features for time series data
def generate_lag_features(input_data, column_name, n_lags=1):
    df = input_data.copy()
    # Create lagged features for the specified number of lags
    for i in range(1, n_lags + 1):
        df[f"{column_name}_lag_{i}"] = df[column_name].shift(i)
    return df.dropna().reset_index(drop=True)

# Function to aggregate data on a weekly basis
def aggregate_weekly(input_data, date_col, target_col):
    df = input_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    # Sum the target column values on a weekly basis
    weekly_df = df[target_col].resample('W').sum().reset_index()
    return weekly_df

# Main function for the deep learning model
def ur2cute_split_deep(input_data, date_col, target_col, lag_features=2, epochs=200, batch_size=32, val_patience=20, split_ratio=0.9, norm=True):
    # Aggregate data weekly and generate lag features
    df = aggregate_weekly(input_data, date_col, target_col)

    # Split the data into training and testing sets
    train_size = int(len(df) * split_ratio)
    train_data = generate_lag_features(df[:train_size], target_col, lag_features)
    test_data = generate_lag_features(df[train_size:], target_col, lag_features)

    # Normalize the features if required
    scaler = MinMaxScaler()
    if norm:
        X_train = scaler.fit_transform(train_data.drop(columns=[target_col]).values)
        X_test = scaler.transform(test_data.drop(columns=[target_col]).values)
    else:
        X_train = train_data.drop(columns=[target_col]).values
        X_test = test_data.drop(columns=[target_col]).values

    # Prepare the target values for training
    y_train = train_data[target_col].values.astype(float)

    # Reshape data for the neural network
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Setup early stopping for training
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=val_patience, restore_best_weights=True)

    # Build and compile the order prediction model
    order_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1), padding="same"),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding="same"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    order_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the order model
    order_model.fit(X_train_reshaped, (train_data[target_col] > 0).astype(int), epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stop])

    # Build and compile the quantity prediction model
    quantity_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1), padding="same"),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding="same"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    custom_optimizer = Adam(learning_rate=0.001)
    # Define a combined loss function (MSE and MAE)
    def combined_loss(alpha=0.5):
        def loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            return alpha * mse + (1 - alpha) * mae
        return loss
    #important note regarding Loss function, here I used combination of mse and mae with an alpha value, you can simply use mse only as loss function and delete the combined loss
    quantity_model.compile(optimizer=custom_optimizer, loss=combined_loss(alpha=0.178))
    # Train the quantity model
    quantity_model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stop])

    # Make predictions using the trained models
    order_predictions = (order_model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
    predicted_quantities = quantity_model.predict(X_test_reshaped).flatten() * order_predictions.astype(float)

    return predicted_quantities
