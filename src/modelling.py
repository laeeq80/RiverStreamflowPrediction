"""Models and scoring for the six-algorithm comparison.

This file builds, trains and scores models. It never reads or writes files:
the experiment scripts pass the data in and decide what to save.

Main entry points
  time_split(X, y)           split in time order into train and test
  compare_models(...)        fit the chosen models, score them on the test part
  get_metrics(obs, sim)      the scores used throughout the project

Every model function has the same call: (X_train, y_train, X_test, seed) and
returns (predictions, settings). Random Forest, XGBoost, SVM and Ridge are
tuned with GridSearchCV on time-ordered folds and ignore the seed. ANN and
LSTM use fixed settings and are run once per seed.

Metrics
  NSE         Nash-Sutcliffe efficiency (same value as R2), 1 is perfect
  RMSE, MAE   in m3/s
  KGE         Kling-Gupta efficiency, 2009 form, 1 is perfect
  PBIAS       100 * sum(obs - sim) / sum(obs); positive = underestimation
  High_RMSE   RMSE on the days when observed discharge is in the top 5%
  High_PBIAS  PBIAS on those same days
"""
import json

import keras
import numpy as np
import pandas as pd
import xgboost as xgb
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import config

METRICS = ["NSE", "RMSE", "MAE", "KGE", "PBIAS", "High_RMSE", "High_PBIAS"]


# ===================== SPLIT =====================
def time_split(X: pd.DataFrame, y: pd.Series,
               train_fraction: float = config.TRAIN_FRACTION):
    """First part of the days for training, the rest for testing."""
    split = int(len(X) * train_fraction)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ===================== METRICS =====================
def nse(obs, sim):
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(obs, sim):
    return 100 * np.sum(obs - sim) / np.sum(obs)


def get_metrics(obs, sim,
                high_flow_quantile: float = config.HIGH_FLOW_QUANTILE) -> dict:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    high = obs >= np.quantile(obs, high_flow_quantile)
    return {
        "NSE": nse(obs, sim),
        "RMSE": np.sqrt(np.mean((obs - sim) ** 2)),
        "MAE": np.mean(np.abs(obs - sim)),
        "KGE": kge(obs, sim),
        "PBIAS": pbias(obs, sim),
        "High_RMSE": np.sqrt(np.mean((obs[high] - sim[high]) ** 2)),
        "High_PBIAS": pbias(obs[high], sim[high]),
    }


# ===================== TUNED MODELS =====================
def _tune(estimator, grid, X_train, y_train, X_test):
    cv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    search = GridSearchCV(estimator, grid, cv=cv, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_.predict(X_test), search.best_params_


def random_forest(X_train, y_train, X_test, seed=None):
    grid = {
        "n_estimators": [100, 160],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    model = RandomForestRegressor(random_state=config.RANDOM_STATE)
    return _tune(model, grid, X_train, y_train, X_test)


def xgboost_model(X_train, y_train, X_test, seed=None):
    grid = {
        "n_estimators": [100, 160],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    }
    model = xgb.XGBRegressor(objective="reg:squarederror",
                             random_state=config.RANDOM_STATE)
    return _tune(model, grid, X_train, y_train, X_test)


def svm(X_train, y_train, X_test, seed=None):
    grid = {
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.1, 0.01],
        "svm__epsilon": [0.1, 0.2],
    }
    pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVR(kernel="rbf"))])
    return _tune(pipe, grid, X_train, y_train, X_test)


def ridge(X_train, y_train, X_test, seed=None):
    grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}
    pipe = Pipeline([("scaler", MinMaxScaler()), ("ridge", Ridge())])
    return _tune(pipe, grid, X_train, y_train, X_test)


# ===================== NEURAL NETWORKS =====================
def _fit_network(model, X_train, y_train):
    # Keras holds out the last 10% of the training rows for early stopping,
    # so the validation days come after the training days in time.
    stop = EarlyStopping(monitor="val_loss", patience=10,
                         restore_best_weights=True)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_split=0.1, epochs=200,
              batch_size=32, callbacks=[stop], verbose=0)
    return model


def ann(X_train, y_train, X_test, seed):
    keras.utils.set_random_seed(seed)
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(X_tr.shape[1],)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    _fit_network(model, X_tr, np.asarray(y_train))
    return model.predict(X_te, verbose=0).flatten(), {}


def lstm(X_train, y_train, X_test, seed):
    # With same-day inputs only, each sample is a single time step.
    # (Sequences of past days are added when lag features are introduced.)
    keras.utils.set_random_seed(seed)
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_train)
    X_te = x_scaler.transform(X_test)
    y_tr = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).flatten()

    X_tr = X_tr.reshape(len(X_tr), 1, -1)
    X_te = X_te.reshape(len(X_te), 1, -1)

    model = Sequential([
        Input(shape=(1, X_tr.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    _fit_network(model, X_tr, y_tr)
    scaled = model.predict(X_te, verbose=0).reshape(-1, 1)
    return y_scaler.inverse_transform(scaled).flatten(), {}


# ===================== MODEL REGISTRY =====================
MODELS = {
    "Random Forest": random_forest,
    "XGBoost": xgboost_model,
    "SVM": svm,
    "Ridge Regression": ridge,
    "ANN": ann,
    "LSTM": lstm,
}
NEURAL_MODELS = ("ANN", "LSTM")      # run once per seed

# short names for the command line
MODEL_ALIASES = {
    "rf": "Random Forest", "xgb": "XGBoost", "svm": "SVM",
    "ridge": "Ridge Regression", "ann": "ANN", "lstm": "LSTM",
}


# ===================== REFERENCE FORECASTS =====================
def seasonal_average(y_train: pd.Series, test_index: pd.DatetimeIndex):
    """Mean discharge of each calendar day in the training years."""
    by_day = y_train.groupby(y_train.index.day_of_year).mean()
    out = pd.Series(test_index.day_of_year, index=test_index).map(by_day)
    return out.fillna(y_train.mean()).values


def yesterday(y: pd.Series, test_index: pd.DatetimeIndex):
    """Yesterday's observed discharge, used as today's forecast."""
    return y.shift(1).loc[test_index].values


# ===================== COMPARISON =====================
def compare_models(X_train, y_train, X_test, y_test, models=None, seeds=None,
                   references=None, verbose=True):
    """Fit each model on the training part and score it on the test part.

    models      names from MODELS (default: all six)
    seeds       seeds for the neural networks (default: config.NN_SEEDS)
    references  optional {name: predictions} scored in the same table
    Returns (runs, predictions): one row per model and seed with all metrics,
    and the test-period predictions of every model next to the observed values.
    """
    models = list(models) if models else list(MODELS)
    seeds = list(seeds) if seeds else list(config.NN_SEEDS)

    runs = []
    predictions = pd.DataFrame({"Observed": y_test})

    def record(name, seed, pred, params=""):
        scores = get_metrics(y_test.values, pred)
        runs.append({"Model": name, "Seed": seed, **scores,
                     "Best_params": params})
        return scores

    for name in models:
        runner = MODELS[name]
        if name in NEURAL_MODELS:
            seed_preds = []
            for seed in seeds:
                if verbose:
                    print(f"[{name}] seed {seed}")
                pred, _ = runner(X_train, y_train, X_test, seed)
                scores = record(name, seed, pred)
                seed_preds.append(pred)
                if verbose:
                    print(f"  NSE {scores['NSE']:.4f} | RMSE {scores['RMSE']:.1f}")
            # the mean over seeds is for the plots and the predictions file only
            predictions[f"{name} (mean of seeds)"] = np.mean(seed_preds, axis=0)
        else:
            if verbose:
                print(f"[{name}]")
            pred, params = runner(X_train, y_train, X_test, None)
            scores = record(name, "", pred, json.dumps(params))
            predictions[name] = pred
            if verbose:
                print(f"  NSE {scores['NSE']:.4f} | RMSE {scores['RMSE']:.1f}")

    for name, pred in (references or {}).items():
        record(name, "", pred)
        predictions[name] = pred

    return pd.DataFrame(runs), predictions
