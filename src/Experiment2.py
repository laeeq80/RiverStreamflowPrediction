"""Experiment 2: which variables should be given as lag features.

Every model gets the same-day inputs (config.BASE_FEATURES) plus the past
`window` days of the variables in one lag combination. Seven combinations are
compared for each of the six models (Random Forest, XGBoost, SVM, Ridge
Regression, ANN, LSTM).

How the choice is made without touching the test years
  1. Validation: for every combination, each model is trained on 2012-2017 and
     scored on 2018-2019. Nothing from 2020-2021 is loaded at this stage.
  2. Selection: for each model, the combination with the highest validation
     NSE is the best one. If other combinations are within the noise of it
     (0.002 NSE, or the spread over seeds if larger) the one with the fewest
     lagged variables is selected.
  3. Test: each model is retrained on 2012-2019 with its selected combination
     and scored once on 2020-2021.

Reads   data/features.csv
Writes  result/experiment2/experiment2_results.xlsx
        result/experiment2/predictions.csv        (test years, selected models)
        result/experiment2/plots/*.png

Examples
  python Experiment2.py
  python Experiment2.py --window 3          # robustness check at another window
  python Experiment2.py --seeds 42 21 --models ridge rf
"""
import argparse
from pathlib import Path

import pandas as pd

import config
import modelling
import reporting
from feature_generation import load_features, make_lags

DEFAULT_WINDOW = 10      # past days given for each lagged variable
TIE_TOLERANCE = 0.002    # NSE difference treated as noise

DISCHARGE = "Water_Discharge"
RAINFALL = "Rainfall"
MAX_TEMP = "Maximum_Temperature"

# The seven lag combinations: name -> variables whose past values are added.
COMBINATIONS = {
    "Discharge": [DISCHARGE],
    "Rainfall": [RAINFALL],
    "MaxTemp": [MAX_TEMP],
    "Rainfall + MaxTemp": [RAINFALL, MAX_TEMP],
    "Discharge + Rainfall": [DISCHARGE, RAINFALL],
    "Discharge + MaxTemp": [DISCHARGE, MAX_TEMP],
    "Discharge + Rainfall + MaxTemp": [DISCHARGE, RAINFALL, MAX_TEMP],
}


def build_inputs(df: pd.DataFrame, variables, window: int):
    """Same-day features plus `window` past days of each variable in `variables`.

    The lags only look backwards (lag_1 is yesterday), and the target never
    appears as a same-day input.
    """
    frame = make_lags(df, {v: window for v in variables})
    lag_columns = [f"{v}_lag_{k}" for v in variables
                   for k in range(1, window + 1)]
    X = frame[config.BASE_FEATURES + lag_columns]
    if config.TARGET in X.columns:
        raise ValueError("The target must not be a same-day input")
    return X, frame[config.TARGET]


# ===================== 1. VALIDATION =====================
def validate_combinations(df, window, models, seeds) -> pd.DataFrame:
    """Score every combination on the validation years (test years excluded)."""
    development = df[df.index <= pd.Timestamp(config.VALIDATION_END)]
    all_runs = []
    for name, variables in COMBINATIONS.items():
        print(f"\n===== validation | {name} | window {window} =====")
        X, y = build_inputs(development, variables, window)
        X_train, X_val, y_train, y_val = modelling.date_split(
            X, y, config.TRAIN_END)
        runs, _ = modelling.compare_models(
            X_train, y_train, X_val, y_val, models=models, seeds=seeds)
        runs.insert(0, "Combination", name)
        all_runs.append(runs)
    return pd.concat(all_runs, ignore_index=True)


# ===================== 2. SELECTION =====================
def choose_combinations(val_summary: pd.DataFrame) -> pd.DataFrame:
    """Best combination per model, and the simplest one within noise of it."""
    rows = []
    for model, group in val_summary.groupby("Model", sort=False):
        best = group.loc[group["NSE"].idxmax()]
        spread = best["NSE_std"] if pd.notna(best["NSE_std"]) else 0.0
        tolerance = max(TIE_TOLERANCE, spread)

        near = group[group["NSE"] >= best["NSE"] - tolerance].copy()
        near["lagged_variables"] = near["Combination"].map(
            lambda c: len(COMBINATIONS[c]))
        pick = near.sort_values(["lagged_variables", "NSE"],
                                ascending=[True, False]).iloc[0]
        rows.append({
            "Model": model,
            "Best combination (validation NSE)": best["Combination"],
            "Best validation NSE": best["NSE"],
            "Selected combination": pick["Combination"],
            "Selected validation NSE": pick["NSE"],
            "Noise tolerance": tolerance,
            "Combinations within noise": len(near),
        })
    return pd.DataFrame(rows)


# ===================== 3. TEST =====================
def test_selected(df, window, selected: dict, seeds):
    """Retrain each model on 2012-2019 with its selected combination and score
    it once on the test years."""
    test_runs, predictions = [], None
    for model, combination in selected.items():
        print(f"\n===== test | {model} | {combination} =====")
        X, y = build_inputs(df, COMBINATIONS[combination], window)
        X_dev, X_test, y_dev, y_test = modelling.date_split(
            X, y, config.VALIDATION_END)
        runs, preds = modelling.compare_models(
            X_dev, y_dev, X_test, y_test, models=[model], seeds=seeds)
        runs.insert(0, "Combination", combination)
        test_runs.append(runs)

        if predictions is None:
            predictions = pd.DataFrame({"Observed": y_test})
            references = {
                "Reference: seasonal average":
                    modelling.seasonal_average(y_dev, y_test.index),
                "Reference: yesterday's discharge":
                    modelling.yesterday(df[config.TARGET], y_test.index),
            }
        for column in preds.columns.drop("Observed"):
            predictions[f"{column} - {combination}"] = preds[column]

    reference_runs = []
    for name, pred in references.items():
        reference_runs.append({
            "Combination": "", "Model": name, "Seed": "",
            **modelling.get_metrics(y_test.values, pred), "Best_params": ""})
        predictions[name] = pred
    test_runs.append(pd.DataFrame(reference_runs))
    return pd.concat(test_runs, ignore_index=True), predictions


# ===================== RUN =====================
def run(features_path: Path = config.FEATURES_PATH,
        out_dir: Path = None,
        window: int = DEFAULT_WINDOW,
        models=None,
        seeds=None):
    if out_dir is None:    # a different window must not overwrite the main run
        out_dir = (config.EXPERIMENT2_DIR if window == DEFAULT_WINDOW else
                   config.EXPERIMENT2_DIR.parent / f"experiment2_window{window}")
    models = list(models) if models else list(modelling.MODELS)
    seeds = list(seeds) if seeds else list(config.NN_SEEDS)
    df = load_features(features_path)
    print(f"window {window} days | train 2012-{config.TRAIN_END[:4]} | "
          f"validation up to {config.VALIDATION_END} | test after that")

    val_runs = validate_combinations(df, window, models, seeds)
    val_summary = reporting.summarise(val_runs, by=["Model", "Combination"])
    choice = choose_combinations(val_summary)
    selected = dict(zip(choice["Model"], choice["Selected combination"]))

    test_runs, predictions = test_selected(df, window, selected, seeds)
    test_summary = reporting.summarise(test_runs, by=["Model", "Combination"])

    # one table: validation choice next to the test score
    test_scores = test_summary.set_index("Model")
    for metric in ("NSE", "RMSE", "MAE", "KGE", "PBIAS"):
        choice[f"Test {metric}"] = choice["Model"].map(test_scores[metric])

    settings = pd.DataFrame({"Setting": [
        "Window (days per lagged variable)", "Same-day inputs", "Train years",
        "Validation years", "Test years", "Neural network seeds",
        "Tie tolerance (NSE)"], "Value": [
        window, ", ".join(config.BASE_FEATURES),
        f"2012 to {config.TRAIN_END[:4]}",
        f"{int(config.TRAIN_END[:4]) + 1} to {config.VALIDATION_END[:4]}",
        f"{int(config.VALIDATION_END[:4]) + 1} to 2021",
        ", ".join(map(str, seeds)), TIE_TOLERANCE]})

    reporting.write_excel(out_dir / "experiment2_results.xlsx", {
        "Best per model": choice,
        "Validation summary": val_summary,
        "Test (selected)": test_summary,
        "Validation runs": val_runs,
        "Test runs": test_runs,
        "Settings": settings,
    })
    predictions.round(3).to_csv(out_dir / "predictions.csv")

    plot_dir = out_dir / "plots"
    table = val_summary.pivot(index="Model", columns="Combination",
                              values="NSE")
    table = table.reindex(index=models, columns=list(COMBINATIONS))
    reporting.plot_score_heatmap(
        table, selected, plot_dir / "validation_nse_heatmap.png",
        title=f"Validation NSE by lag combination (window {window} days); "
              "outlined = selected",
        colorbar_label="Validation NSE (2018-2019)")
    reporting.plot_predictions(predictions, plot_dir)

    print("\nValidation NSE (mean over seeds):")
    print(table.round(3).to_string())
    print("\nBest lag combination for each model:")
    print(choice[["Model", "Best combination (validation NSE)",
                  "Selected combination", "Test NSE"]]
          .round(3).to_string(index=False))
    print(f"\nResults saved to {out_dir}")
    return choice


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"past days per lagged variable (default {DEFAULT_WINDOW})")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help=f"seeds for ANN and LSTM (default {config.NN_SEEDS})")
    parser.add_argument("--models", nargs="+",
                        choices=sorted(modelling.MODEL_ALIASES),
                        help="models to run (default: all six)")
    parser.add_argument("--out-dir", type=Path,
                        help="folder for the results (default result/experiment2)")
    args = parser.parse_args(argv)

    models = ([modelling.MODEL_ALIASES[m] for m in args.models]
              if args.models else None)
    run(out_dir=args.out_dir, window=args.window, models=models,
        seeds=args.seeds)


if __name__ == "__main__":
    main()
