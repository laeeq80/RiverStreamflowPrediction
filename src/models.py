import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
df=pd.read_csv(r'../data/processed/process.csv')
x=df.drop(columns=['water_Discharge','Unnamed: 0'])
y=df['water_Discharge']

def train_random_forest(x, y, test_size=0.2, n_estimators=160, random_state=42):
    """
    Trains a Random Forest Regressor, evaluates it, and returns the trained model & scores.
    
    Parameters:
        x (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target variable.
        test_size (float): Fraction of the data to use for testing.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        model (RandomForestRegressor): The trained Random Forest model.
        mse (float): Mean Squared Error on test data.
        r2 (float): R² score on test data.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Train model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error of random_forest: {mse}")
    print(f"R-squared Score of random_forest: {r2}")

    return rf_model, mse, r2
print(train_random_forest(x,y,n_estimators=160,test_size=0.2,random_state=42))




# xgboost model

def train_xgboost_regressor(x, y, test_size=0.2, random_state=42):
    """
    Trains an XGBoost Regressor with hyperparameter tuning and plots predictions vs actual values.
    
    Parameters:
        x (pd.DataFrame or np.array): Features.
        y (pd.Series or np.array): Target.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        dict: Best parameters, Mean Squared Error, R² Score, trained model.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define XGBoost model
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 160],
        'max_depth': [2, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Grid search
    grid_search = GridSearchCV(
        xgb_regressor, 
        param_grid, 
        cv=10, 
        scoring='neg_mean_squared_error', 
        verbose=1, 
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "best_params": grid_search.best_params_,
        "mse of xgboost": mse,
        "r2 of xgboost": r2,
        "model": best_model
    }
print(train_xgboost_regressor(x,y,0.2,42))

#polynomial

def train_polynomial_regression(X, y, degree=3, test_size=0.2, random_state=42):
    """
    Train and evaluate a Polynomial Regression model without plotting or try/except.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Polynomial Degree: {degree}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

    return model, {"MSE of plynial": mse, "RMSE": rmse, "R² of plynomial": r2}
print(train_polynomial_regression(x,y))

# SVR
def train_svr_rbf(X, y, test_size=0.2, random_state=42):
    """
    Train an SVR (RBF kernel) using GridSearchCV with train-test split inside.
    Automatically scales features.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature scaling (important for SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Parameter grid for GridSearch
    param_grid = {
        'C': [1, 10, 100],           
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'epsilon': [0.1, 0.5, 1]     
    }

    # Initialize SVR model
    svr = SVR(kernel='rbf')

    # GridSearchCV for best hyperparameters
    grid_search = GridSearchCV(
        svr, param_grid,
        cv=6,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    # Best model
    best_svr = grid_search.best_estimator_

    # Predictions
    y_pred = best_svr.predict(X_test_scaled)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Best Parameters:", grid_search.best_params_)
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

    return best_svr, {"MSE of svr": mse, "RMSE": rmse, "R² svr": r2}
print(train_svr_rbf(x,y))

# to train the ANN 

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def train_ann_model(df, window_size=20, n_features=4, test_size=0.2, epochs=50, batch_size=64, random_state=42):
    """
    Trains an Artificial Neural Network (ANN) to predict water discharge using time-series data.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with columns [rainfall, max_temp, min_temp, avg_temp, water_discharge].
        window_size (int): Number of past days to consider for sequence creation (default: 20).
        n_features (int): Number of feature columns (default: 4, rainfall + 3 temperatures).
        test_size (float): Fraction of data for testing (default: 0.2).
        epochs (int): Number of training epochs (default: 50).
        batch_size (int): Batch size for training (default: 64).
        random_state (int): Random seed for reproducibility (default: 42).
    
    Returns:
        model (Sequential): Trained ANN model.
        y_test_orig (np.ndarray): Original-scale true test values.
        y_pred_orig (np.ndarray): Original-scale predicted values.
        history (History): Training history for plotting.
    """
    # 1. Load and preprocess data
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input df must be a non-empty pandas DataFrame.")
    if not all(col in df.columns for col in ['rainfall', 'max_temp', 'min_temp', 'avg_temp', 'water_discharge']):
        raise ValueError("DataFrame must contain columns: rainfall, max_temp, min_temp, avg_temp, water_discharge.")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # 2. Create sequences with windowing
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size, :-1].flatten())  # Flatten 20 days of features (4 features)
        y.append(scaled_data[i + window_size, -1])            # Corresponding discharge value

    X = np.array(X)
    y = np.array(y)

    # 3. Train-test split (time-series aware, no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    # 4. Build deeper ANN model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for discharge prediction
    ])

    # 5. Compile and train
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=30, restore_best_weights=True)],
        verbose=1
    )

    # 6. Evaluate
    y_pred = model.predict(X_test)

    # Inverse scaling for discharge
    dummy_matrix = np.zeros((len(y_pred), scaled_data.shape[1]))
    dummy_matrix[:, -1] = y_pred.flatten()
    y_pred_orig = scaler.inverse_transform(dummy_matrix)[:, -1]

    # True values inverse scaling
    dummy_test = np.zeros((len(y_test), scaled_data.shape[1]))
    dummy_test[:, -1] = y_test
    y_test_orig = scaler.inverse_transform(dummy_test)[:, -1]

    # Print metrics
    r2 = r2_score(y_test_orig, y_pred_orig)
    rmse = mean_squared_error(y_test_orig, y_pred_orig)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} m³/s")

    return model, y_test_orig, y_pred_orig, history




#LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_time_series(df, target_col, window_size=20, test_size=0.2, epochs=20, batch_size=32, lstm_units=[64, 32]):
    """
    Train an LSTM model for time-series forecasting and evaluate its performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing features and target column
    target_col : str
        Name of the target column in the DataFrame
    window_size : int, optional
        Number of time steps for sequence creation (default: 20)
    test_size : float, optional
        Proportion of data for testing (default: 0.2)
    epochs : int, optional
        Number of training epochs (default: 20)
    batch_size : int, optional
        Batch size for training (default: 32)
    lstm_units : list, optional
        List of LSTM layer units (default: [64, 32])
    
    Returns:
    --------
    dict : Contains model, history, metrics, and predictions
    """
    # Ensure target column is the last column
    cols = [col for col in df.columns if col != target_col] + [target_col]
    df = df[cols]
    
    # Feature scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Sequence preparation
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size, :-1])  # All features except target
            y.append(data[i + window_size, -1])     # Target
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_data, window_size)
    
    # Train-test split (no shuffle for time-series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Build LSTM model
    model = Sequential()
    for i, units in enumerate(lstm_units):
        if i == 0:
            model.add(LSTM(units, return_sequences=(len(lstm_units) > 1), input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            model.add(LSTM(units, return_sequences=(i < len(lstm_units) - 1)))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    early_stop = EarlyStopping(patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Inverse scaling
    def inverse_target(scaled_y):
        dummy = np.zeros((len(scaled_y), scaled_data.shape[1]))
        dummy[:, -1] = scaled_y.flatten()
        return scaler.inverse_transform(dummy)[:, -1]
    
    y_train_orig = inverse_target(y_train)
    y_train_pred_orig = inverse_target(y_train_pred)
    y_test_orig = inverse_target(y_test)
    y_test_pred_orig = inverse_target(y_test_pred)
    
    # Evaluate
    train_r2 = r2_score(y_train_orig, y_train_pred_orig)
    test_r2 = r2_score(y_test_orig, y_test_pred_orig)
    train_rmse = mean_squared_error(y_train_orig, y_train_pred_orig, squared=False)
    test_rmse = mean_squared_error(y_test_orig, y_test_pred_orig, squared=False)
    
    # Print results
    print("\n📊 Training Results")
    print(f"R² Score: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    
    print("\n📊 Testing Results")
    print(f"R² Score: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    # Return results
    results = {
        'model': model,
        'history': history,
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        },
        'predictions': {
            'y_train_orig': y_train_orig,
            'y_train_pred_orig': y_train_pred_orig,
            'y_test_orig': y_test_orig,
            'y_test_pred_orig': y_test_pred_orig
        },
        'scaler': scaler
    }
    
    return results

# Example usage:
# df = pd.DataFrame(...)  # Your DataFrame with features and target
# results = train_lstm_time_series(df, target_col='water_discharge', window_size=20, test_size=0.2, epochs=20, batch_size=32)