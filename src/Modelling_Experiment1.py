"""Experiment 1: six models without lag features.

Compares Random Forest, XGBoost, SVM, Ridge Regression, ANN and LSTM using
only same-day inputs (the seven features in feature_generation.BASE_FEATURES)
to predict Water_Discharge.

Reads   data/features.csv
Writes  result/experiment1/experiment1_results.xlsx   (summary + every run)
        result/experiment1/predictions.csv            (test period, all models)
        result/experiment1/plots/*.png

Split: first 80% of days for training, last 20% for testing, in time order.
Random Forest, XGBoost, SVM and Ridge are tuned with GridSearchCV on the
training part using time-ordered folds. ANN and LSTM use fixed settings and
are run with several seeds; their results are reported as mean and std.
The LSTM here sees one time step (the same day), so it works as a static
regressor, as in the other five models.

Metrics (test period)
  NSE         Nash-Sutcliffe efficiency (same value as R2), 1 is perfect
  RMSE, MAE   in m3/s
  KGE         Kling-Gupta efficiency, 2009 form, 1 is perfect
  PBIAS       100 * sum(obs - sim) / sum(obs); positive = underestimation
  High_RMSE   RMSE on the days when observed discharge is in the top 5%
  High_PBIAS  PBIAS on those same days
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import keras
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential

from feature_generation import BASE_FEATURES, FEATURES_PATH, TARGET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "result" / "experiment1"
PLOT_DIR = OUT_DIR / "plots"

TRAIN_FRACTION = 0.8
HIGH_FLOW_QUANTILE = 0.95
CV = TimeSeriesSplit(n_splits=5)
RANDOM_STATE = 42
NN_SEEDS = [42, 21, 256, 512, 1024]

METRICS = ["NSE", "RMSE", "MAE", "KGE", "PBIAS", "High_RMSE", "High_PBIAS"]


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


def get_metrics(obs, sim) -> dict:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    high = obs >= np.quantile(obs, HIGH_FLOW_QUANTILE)
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
def tune(estimator, grid, X_train, y_train, X_test):
    search = GridSearchCV(estimator, grid, cv=CV, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_.predict(X_test), search.best_params_


def run_random_forest(X_train, y_train, X_test):
    grid = {
        "n_estimators": [100, 160],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    return tune(RandomForestRegressor(random_state=RANDOM_STATE), grid,
                X_train, y_train, X_test)


def run_xgboost(X_train, y_train, X_test):
    grid = {
        "n_estimators": [100, 160],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    }
    model = xgb.XGBRegressor(objective="reg:squarederror",
                             random_state=RANDOM_STATE)
    return tune(model, grid, X_train, y_train, X_test)


def run_svm(X_train, y_train, X_test):
    grid = {
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.1, 0.01],
        "svm__epsilon": [0.1, 0.2],
    }
    pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVR(kernel="rbf"))])
    return tune(pipe, grid, X_train, y_train, X_test)


def run_ridge(X_train, y_train, X_test):
    grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}
    pipe = Pipeline([("scaler", MinMaxScaler()), ("ridge", Ridge())])
    return tune(pipe, grid, X_train, y_train, X_test)


# ===================== NEURAL NETWORKS =====================
def fit_network(model, X_train, y_train):
    # Keras holds out the last 10% of the training rows for early stopping,
    # so the validation days come after the training days in time.
    stop = EarlyStopping(monitor="val_loss", patience=10,
                         restore_best_weights=True)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_split=0.1, epochs=200,
              batch_size=32, callbacks=[stop], verbose=0)
    return model


def run_ann(X_train, y_train, X_test, seed):
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
    fit_network(model, X_tr, np.asarray(y_train))
    return model.predict(X_te, verbose=0).flatten()


def run_lstm(X_train, y_train, X_test, seed):
    keras.utils.set_random_seed(seed)
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_train)
    X_te = x_scaler.transform(X_test)
    y_tr = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).flatten()

    # one time step per sample: the features of the day being predicted
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
    fit_network(model, X_tr, y_tr)
    scaled = model.predict(X_te, verbose=0).reshape(-1, 1)
    return y_scaler.inverse_transform(scaled).flatten()


# ===================== REFERENCE FORECASTS =====================
def seasonal_average(y_train, test_index):
    """Mean discharge of each calendar day in the training years."""
    by_day = y_train.groupby(y_train.index.day_of_year).mean()
    out = pd.Series(test_index.day_of_year, index=test_index).map(by_day)
    return out.fillna(y_train.mean()).values


# ===================== PLOTS =====================
def plot_predictions(obs: pd.Series, pred: np.ndarray, name: str, m: dict):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(obs.index, obs.values, label="Observed", linewidth=1)
    ax.plot(obs.index, pred, label="Predicted", linewidth=1, linestyle="--")
    ax.set_title(f"{name} | NSE {m['NSE']:.3f} | RMSE {m['RMSE']:.1f} m³/s")
    ax.set_xlabel("Date")
    ax.set_ylabel("Water discharge (m³/s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    file = name.lower().replace(" ", "_") + ".png"
    fig.savefig(PLOT_DIR / file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================== MAIN =====================
def main(seeds=NN_SEEDS) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"], index_col="Date")
    X, y = df[BASE_FEATURES], df[TARGET]
    split = int(len(df) * TRAIN_FRACTION)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"{len(BASE_FEATURES)} features | train {len(X_train)} days "
          f"({X_train.index[0].date()} to {X_train.index[-1].date()}) | "
          f"test {len(X_test)} days ({X_test.index[0].date()} to "
          f"{X_test.index[-1].date()})")

    runs = []                        # one row per model and seed
    predictions = pd.DataFrame({"Observed": y_test})

    def record(name, seed, pred, params=""):
        m = get_metrics(y_test.values, pred)
        runs.append({"Model": name, "Seed": seed, **m, "Best_params": params})
        return m

    # tuned models: no randomness left, so one run each
    for name, runner in [("Random Forest", run_random_forest),
                         ("XGBoost", run_xgboost),
                         ("SVM", run_svm),
                         ("Ridge Regression", run_ridge)]:
        print(f"[{name}]")
        pred, params = runner(X_train, y_train, X_test)
        m = record(name, "", pred, json.dumps(params))
        predictions[name] = pred
        plot_predictions(y_test, pred, name, m)
        print(f"  NSE {m['NSE']:.4f} | RMSE {m['RMSE']:.1f}")

    # neural networks: one run per seed
    for name, runner in [("ANN", run_ann), ("LSTM", run_lstm)]:
        seed_preds = []
        for seed in seeds:
            print(f"[{name}] seed {seed}")
            pred = runner(X_train, y_train, X_test, seed)
            m = record(name, seed, pred)
            seed_preds.append(pred)
            print(f"  NSE {m['NSE']:.4f} | RMSE {m['RMSE']:.1f}")
        mean_pred = np.mean(seed_preds, axis=0)   # for the plot and file only
        predictions[name + " (mean of seeds)"] = mean_pred
        plot_predictions(y_test, mean_pred, name + " (mean of seeds)",
                         get_metrics(y_test.values, mean_pred))

    # reference forecasts (not models in the comparison)
    for name, pred in [
        ("Reference: seasonal average", seasonal_average(y_train, y_test.index)),
        ("Reference: yesterday's discharge", y.shift(1).loc[y_test.index].values),
    ]:
        record(name, "", pred)
        predictions[name] = pred

    runs = pd.DataFrame(runs)
    grouped = runs.groupby("Model", sort=False)[METRICS]
    summary = grouped.mean().join(grouped.std().add_suffix("_std"))
    summary.insert(0, "Runs", runs.groupby("Model", sort=False).size())
    summary = summary.reset_index()

    with pd.ExcelWriter(OUT_DIR / "experiment1_results.xlsx") as writer:
        summary.round(4).to_excel(writer, sheet_name="Summary", index=False)
        runs.round(4).to_excel(writer, sheet_name="All runs", index=False)
    predictions.round(3).to_csv(OUT_DIR / "predictions.csv")

    print("\n" + summary[["Model", "Runs", "NSE", "RMSE", "MAE", "KGE",
                          "PBIAS", "High_RMSE", "High_PBIAS"]]
          .round(3).to_string(index=False))
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()