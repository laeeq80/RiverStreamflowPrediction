"""Experiment 1: six models without lag features.

Compares Random Forest, XGBoost, SVM, Ridge Regression, ANN and LSTM using
only same-day inputs (config.BASE_FEATURES) to predict Water_Discharge.

Reads   data/features.csv
Writes  result/experiment1/experiment1_results.xlsx   (summary + every run)
        result/experiment1/predictions.csv            (test period, all models)
        result/experiment1/plots/*.png

The models and scores live in modelling.py, the tables and figures in
reporting.py. This file chooses the data and settings and connects them.

Examples
  python Experiment1.py
  python Experiment1.py --seeds 42 21 --models ridge rf
"""
import argparse
from pathlib import Path

import config
import modelling
import reporting
from feature_generation import load_features


def run(features_path: Path = config.FEATURES_PATH,
        out_dir: Path = config.EXPERIMENT1_DIR,
        features=None,
        models=None,
        seeds=None,
        train_fraction: float = config.TRAIN_FRACTION):
    features = list(features) if features else list(config.BASE_FEATURES)

    df = load_features(features_path)
    X, y = df[features], df[config.TARGET]
    X_train, X_test, y_train, y_test = modelling.time_split(X, y, train_fraction)
    print(f"{len(features)} features | "
          f"train {len(X_train)} days ({X_train.index[0].date()} to "
          f"{X_train.index[-1].date()}) | "
          f"test {len(X_test)} days ({X_test.index[0].date()} to "
          f"{X_test.index[-1].date()})")

    # reference forecasts: not models in the comparison, only yardsticks
    references = {
        "Reference: seasonal average":
            modelling.seasonal_average(y_train, y_test.index),
        "Reference: yesterday's discharge":
            modelling.yesterday(y, y_test.index),
    }

    runs, predictions = modelling.compare_models(
        X_train, y_train, X_test, y_test,
        models=models, seeds=seeds, references=references)

    summary = reporting.summarise(runs)
    reporting.save_results(out_dir, runs, summary, predictions,
                           excel_name="experiment1_results.xlsx")
    reporting.print_summary(summary)
    print(f"\nResults saved to {out_dir}")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+",
                        help=f"seeds for ANN and LSTM (default {config.NN_SEEDS})")
    parser.add_argument("--models", nargs="+",
                        choices=sorted(modelling.MODEL_ALIASES),
                        help="models to run (default: all six)")
    parser.add_argument("--train-fraction", type=float,
                        default=config.TRAIN_FRACTION,
                        help="share of days used for training "
                             f"(default {config.TRAIN_FRACTION})")
    args = parser.parse_args(argv)

    models = ([modelling.MODEL_ALIASES[m] for m in args.models]
              if args.models else None)
    run(models=models, seeds=args.seeds, train_fraction=args.train_fraction)


if __name__ == "__main__":
    main()
