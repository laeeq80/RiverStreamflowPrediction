"""Experiment 1: six models without lag features.

Compares Random Forest, XGBoost, SVM, Ridge Regression, ANN and LSTM using
only same-day inputs (config.BASE_FEATURES) to predict Water_Discharge.

Nothing is chosen in this experiment, but it is scored the same way as
Experiments 2 and 3:
  Validation  train 2012-2017, score 2018-2019
  Test        train 2012-2019, score 2020-2021
The test years are never used to choose anything.

Reads   data/features.csv
Writes  result/experiment1/experiment1_results.xlsx
          (Validation summary, Test summary, Validation runs, Test runs, Settings)
        result/experiment1/predictions.csv     (test years, all models)
        result/experiment1/plots/*.png

The models and scores live in modelling.py, the tables and figures in
reporting.py. This file chooses the data and settings and connects them.

Examples
  python Experiment1.py
  python Experiment1.py --seeds 42 21 --models ridge rf
"""
import argparse
from pathlib import Path

import pandas as pd

import config
import modelling
import reporting
from feature_generation import load_features


def score(X_train, y_train, X_test, y_test, y_all, models, seeds):
    """Train on one period, score on the next, with the two reference forecasts."""
    references = {
        "Reference: seasonal average":
            modelling.seasonal_average(y_train, y_test.index),
        "Reference: yesterday's discharge":
            modelling.yesterday(y_all, y_test.index),
    }
    return modelling.compare_models(
        X_train, y_train, X_test, y_test,
        models=models, seeds=seeds, references=references)


def describe(label, X_train, X_test):
    print(f"{label}: train {len(X_train)} days "
          f"({X_train.index[0].date()} to {X_train.index[-1].date()}) | "
          f"score {len(X_test)} days "
          f"({X_test.index[0].date()} to {X_test.index[-1].date()})")


def run(features_path: Path = config.FEATURES_PATH,
        out_dir: Path = config.EXPERIMENT1_DIR,
        features=None,
        models=None,
        seeds=None):
    features = list(features) if features else list(config.BASE_FEATURES)
    seeds_used = list(seeds) if seeds else list(config.NN_SEEDS)

    df = load_features(features_path)
    X, y = df[features], df[config.TARGET]
    print(f"{len(features)} features")

    # ---- validation: train 2012-2017, score 2018-2019 (test years left out)
    development = df.index <= pd.Timestamp(config.VALIDATION_END)
    X_train, X_val, y_train, y_val = modelling.date_split(
        X[development], y[development], config.TRAIN_END)
    describe("Validation", X_train, X_val)
    val_runs, _ = score(X_train, y_train, X_val, y_val, y, models, seeds)

    # ---- test: train 2012-2019, score 2020-2021
    X_dev, X_test, y_dev, y_test = modelling.date_split(
        X, y, config.VALIDATION_END)
    describe("Test", X_dev, X_test)
    test_runs, predictions = score(X_dev, y_dev, X_test, y_test, y, models, seeds)

    val_summary = reporting.summarise(val_runs)
    test_summary = reporting.summarise(test_runs)

    settings = pd.DataFrame({"Setting": [
        "Same-day inputs", "Lag features", "Train years (validation)",
        "Validation years", "Train years (test)", "Test years",
        "Neural network seeds"], "Value": [
        ", ".join(features), "none",
        f"2012 to {config.TRAIN_END[:4]}",
        f"{int(config.TRAIN_END[:4]) + 1} to {config.VALIDATION_END[:4]}",
        f"2012 to {config.VALIDATION_END[:4]}",
        f"{int(config.VALIDATION_END[:4]) + 1} to 2021",
        ", ".join(map(str, seeds_used))]})

    reporting.write_excel(out_dir / "experiment1_results.xlsx", {
        "Validation summary": val_summary,
        "Test summary": test_summary,
        "Validation runs": val_runs,
        "Test runs": test_runs,
        "Settings": settings,
    })
    predictions.round(3).to_csv(out_dir / "predictions.csv")
    reporting.plot_predictions(predictions, out_dir / "plots")

    print("\nValidation (2018-2019):")
    reporting.print_summary(val_summary)
    print("\nTest (2020-2021):")
    reporting.print_summary(test_summary)
    print(f"\nResults saved to {out_dir}")
    return test_summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+",
                        help=f"seeds for ANN and LSTM (default {config.NN_SEEDS})")
    parser.add_argument("--models", nargs="+",
                        choices=sorted(modelling.MODEL_ALIASES),
                        help="models to run (default: all six)")
    args = parser.parse_args(argv)

    models = ([modelling.MODEL_ALIASES[m] for m in args.models]
              if args.models else None)
    run(models=models, seeds=args.seeds)


if __name__ == "__main__":
    main()