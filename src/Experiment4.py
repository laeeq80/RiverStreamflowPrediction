"""Experiment 4: predicting several days ahead.

The lead time is the number of days between the last observed day and the day
that is predicted. With a lead time of 3 days and lags ending on 17 March, the
prediction is for 20 March. Lead time 1 (the previous experiments) is not
repeated here.

Every model keeps what Experiments 2 and 3 selected for it: its lag combination
and its window (read from result/experiment3/experiment3_results.xlsx, sheet
"Best per model"). Nothing is chosen in this experiment, so, as in Experiment 1,
every model is scored twice:
  Validation  train 2012-2017, score 2018-2019
  Test        train 2012-2019, score 2020-2021
The test years are never used to choose anything.

Inputs for a lead time of h days
  - the past `window` days of each variable in the model's combination, ending
    h days before the predicted day (column <variable>_lag_1 is the last
    observed day, so it is h days before the target, _lag_2 the day before
    that, and so on)
  - the day-of-year sin and cos, which are known in advance
  The same-day rainfall and temperature of the predicted day are NOT used,
  because they are not known h days earlier.

References (scored in the same tables)
  seasonal average   mean discharge of each calendar day in the training years
  persistence        the discharge observed h days before the predicted day

Reads   data/features.csv
        result/experiment3/experiment3_results.xlsx
Writes  result/experiment4/experiment4_results.xlsx
        result/experiment4/predictions.csv        (test years, all horizons)
        result/experiment4/plots/*.png

Examples
  python Experiment4.py
  python Experiment4.py --horizons 3 7
  python Experiment4.py --seeds 42 21 --models ridge rf
"""
import argparse
from pathlib import Path

import pandas as pd

import config
import modelling
import reporting
from Experiment2 import COMBINATIONS
from feature_generation import check_daily, load_features

DEFAULT_HORIZONS = [3, 5, 7]          # days between last observed day and target
DEFAULT_OUT_DIR = config.RESULT_DIR / "experiment4"
EXPERIMENT3_RESULTS = config.RESULT_DIR / "experiment3" / "experiment3_results.xlsx"

# inputs that are known in advance, whatever the lead time
CALENDAR_FEATURES = [c for c in config.BASE_FEATURES if c.startswith("DayOfYear")]
SEASONAL = "Reference: seasonal average"
PERSISTENCE = "Reference: persistence"


def load_settings(path: Path, models) -> pd.DataFrame:
    """Each model's combination and window from the Experiment 3 results."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run Experiment3.py first (or pass --exp3-results).")
    table = pd.read_excel(path, sheet_name="Best per model")
    table = table.set_index("Model")
    rows = []
    for model in models:
        if model not in table.index:
            raise ValueError(f"{model} is not in the Experiment 3 results")
        combination = table.loc[model, "Combination (Experiment 2)"]
        if combination not in COMBINATIONS:
            raise ValueError(f"Unknown combination: {combination}")
        rows.append({"Model": model, "Combination": combination,
                     "Window": int(table.loc[model, "Selected window"])})
    return pd.DataFrame(rows)


def build_inputs(df: pd.DataFrame, variables, window: int, horizon: int):
    """Inputs and target for predicting `horizon` days after the last observed day.

    <variable>_lag_j holds the value of the variable (horizon + j - 1) days
    before the predicted day, for j = 1..window. The columns are named like the
    ones of the earlier experiments, so the models read them in the same way.
    """
    check_daily(df)
    columns = {f"{v}_lag_{j}": df[v].shift(horizon + j - 1)
               for v in variables for j in range(1, window + 1)}
    X = pd.concat([df[CALENDAR_FEATURES], pd.DataFrame(columns, index=df.index)],
                  axis=1)
    first = horizon + window - 1          # rows before this have no history
    X, y = X.iloc[first:], df[config.TARGET].iloc[first:]
    if config.TARGET in X.columns:
        raise ValueError("The target must not be a same-day input")
    return X, y


def references(target: pd.Series, train_end, index, horizon: int) -> dict:
    """Reference forecasts for the scored days. They do not depend on the model."""
    train = target[target.index <= pd.Timestamp(train_end)]
    return {
        SEASONAL: modelling.seasonal_average(train, index),
        PERSISTENCE: target.shift(horizon).loc[index].values,
    }


def reference_rows(horizon, observed, refs) -> pd.DataFrame:
    return pd.DataFrame([{
        "Horizon": horizon, "Combination": "", "Window": "", "Model": name,
        "Seed": "", **modelling.get_metrics(observed.values, pred),
        "Best_params": ""} for name, pred in refs.items()])


def run(features_path: Path = config.FEATURES_PATH,
        exp3_results: Path = EXPERIMENT3_RESULTS,
        out_dir: Path = DEFAULT_OUT_DIR,
        horizons=None,
        models=None,
        seeds=None):
    horizons = sorted(horizons) if horizons else list(DEFAULT_HORIZONS)
    models = list(models) if models else list(modelling.MODELS)
    seeds = list(seeds) if seeds else list(config.NN_SEEDS)

    used = load_settings(exp3_results, models)
    print("Combination and window of each model (from Experiment 3):")
    print(used.to_string(index=False))
    df = load_features(features_path)
    target = df[config.TARGET]
    print(f"lead times {horizons} days | validation: train 2012-"
          f"{config.TRAIN_END[:4]}, score up to {config.VALIDATION_END[:4]} | "
          f"test: train up to {config.VALIDATION_END[:4]}, score after that")

    val_runs, test_runs, test_predictions = [], [], {}
    for horizon in horizons:
        print(f"\n===== lead time {horizon} days =====")
        for _, row in used.iterrows():
            model, combination, window = row["Model"], row["Combination"], row["Window"]
            print(f"\n----- {model} | {combination} | window {window} | "
                  f"{horizon} days ahead -----")
            X, y = build_inputs(df, COMBINATIONS[combination], window, horizon)

            # validation: train 2012-2017, score 2018-2019 (test years left out)
            development = X.index <= pd.Timestamp(config.VALIDATION_END)
            X_train, X_val, y_train, y_val = modelling.date_split(
                X[development], y[development], config.TRAIN_END)
            runs, _ = modelling.compare_models(
                X_train, y_train, X_val, y_val, models=[model], seeds=seeds)
            val_index = y_val.index

            # test: train 2012-2019, score 2020-2021
            X_dev, X_test, y_dev, y_test = modelling.date_split(
                X, y, config.VALIDATION_END)
            test, preds = modelling.compare_models(
                X_dev, y_dev, X_test, y_test, models=[model], seeds=seeds)
            test_index = y_test.index

            for frame, bucket in ((runs, val_runs), (test, test_runs)):
                frame.insert(0, "Window", window)
                frame.insert(0, "Combination", combination)
                frame.insert(0, "Horizon", horizon)
                bucket.append(frame)
            test_predictions.setdefault(horizon, pd.DataFrame(
                {"Observed": y_test}))
            for column in preds.columns.drop("Observed"):
                test_predictions[horizon][column] = preds[column]

        # reference forecasts of this lead time, added once per stage
        val_observed = target.loc[val_index]
        test_observed = target.loc[test_index]
        val_refs = references(target, config.TRAIN_END, val_index, horizon)
        test_refs = references(target, config.VALIDATION_END, test_index, horizon)
        val_runs.append(reference_rows(horizon, val_observed, val_refs))
        test_runs.append(reference_rows(horizon, test_observed, test_refs))
        for name, pred in test_refs.items():
            test_predictions[horizon][name] = pred

    val_runs = pd.concat(val_runs, ignore_index=True)
    test_runs = pd.concat(test_runs, ignore_index=True)
    val_summary = reporting.summarise(val_runs, by=["Model", "Horizon"])
    test_summary = reporting.summarise(test_runs, by=["Model", "Horizon"])
    for summary in (val_summary, test_summary):
        summary.insert(2, "Window", summary["Model"].map(
            dict(zip(used["Model"], used["Window"]))).fillna(""))
        summary.insert(2, "Combination", summary["Model"].map(
            dict(zip(used["Model"], used["Combination"]))).fillna(""))

    order = models + [SEASONAL, PERSISTENCE]

    def nse_table(summary):
        table = summary.pivot(index="Model", columns="Horizon", values="NSE")
        table = table.reindex(index=order, columns=horizons)
        table.columns = [f"{h} days ahead" for h in horizons]
        return table

    val_table, test_table = nse_table(val_summary), nse_table(test_summary)

    settings = pd.DataFrame({"Setting": [
        "Lead times (days between last observed day and target)",
        "Same-day inputs", "Train years (validation)", "Validation years",
        "Train years (test)", "Test years", "Neural network seeds",
        "Experiment 3 results used"], "Value": [
        ", ".join(map(str, horizons)), ", ".join(CALENDAR_FEATURES),
        f"2012 to {config.TRAIN_END[:4]}",
        f"{int(config.TRAIN_END[:4]) + 1} to {config.VALIDATION_END[:4]}",
        f"2012 to {config.VALIDATION_END[:4]}",
        f"{int(config.VALIDATION_END[:4]) + 1} to 2021",
        ", ".join(map(str, seeds)), str(exp3_results)]})

    reporting.write_excel(out_dir / "experiment4_results.xlsx", {
        "Validation summary": val_summary,
        "Test summary": test_summary,
        "Validation NSE": val_table.reset_index().rename(
            columns={"index": "Model"}),
        "Test NSE": test_table.reset_index().rename(columns={"index": "Model"}),
        "Validation runs": val_runs,
        "Test runs": test_runs,
        "Models used": used,
        "Settings": settings,
    })

    combined = pd.DataFrame({"Observed": next(iter(
        test_predictions.values()))["Observed"]})
    for horizon, frame in test_predictions.items():
        for column in frame.columns.drop("Observed"):
            combined[f"{column} - {horizon} days ahead"] = frame[column]
    combined.round(3).to_csv(out_dir / "predictions.csv")

    plot_dir = out_dir / "plots"
    for horizon, frame in test_predictions.items():
        reporting.plot_predictions(frame, plot_dir / f"{horizon}_days_ahead")
    reporting.plot_score_heatmap(
        val_table.loc[models], {}, plot_dir / "validation_nse_heatmap.png",
        title="Validation NSE by lead time",
        colorbar_label="Validation NSE (2018-2019)")
    reporting.plot_score_heatmap(
        test_table.loc[models], {}, plot_dir / "test_nse_heatmap.png",
        title="Test NSE by lead time",
        colorbar_label="Test NSE (2020-2021)")

    print("\nValidation NSE (mean over seeds):")
    print(val_table.round(3).to_string())
    print("\nTest NSE (mean over seeds):")
    print(test_table.round(3).to_string())
    print(f"\nResults saved to {out_dir}")
    return test_summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--horizons", type=int, nargs="+",
                        help=f"lead times in days (default {DEFAULT_HORIZONS})")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help=f"seeds for RF, XGBoost, ANN and LSTM (default {config.NN_SEEDS})")
    parser.add_argument("--models", nargs="+",
                        choices=sorted(modelling.MODEL_ALIASES),
                        help="models to run (default: all six)")
    parser.add_argument("--exp3-results", type=Path, default=EXPERIMENT3_RESULTS,
                        help="Experiment 3 results file to read the settings from")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="folder for the results (default result/experiment4)")
    args = parser.parse_args(argv)

    models = ([modelling.MODEL_ALIASES[m] for m in args.models]
              if args.models else None)
    run(exp3_results=args.exp3_results, out_dir=args.out_dir,
        horizons=args.horizons, models=models, seeds=args.seeds)


if __name__ == "__main__":
    main()