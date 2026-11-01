"""Experiment 3: how many past days (lag window) to give each model.

Every model keeps the lag combination it selected in Experiment 2 (read from
result/experiment2/experiment2_results.xlsx, sheet "Best per model") and is
run with each window length. Six models x five windows = 30 model-window pairs.

The choice is made the same way as in Experiments 1 and 2, without the test years
  1. Validation: for every pair, the model is trained on 2012-2017 and scored
     on 2018-2019. Nothing from 2020-2021 is loaded at this stage.
  2. Selection: for each model, the window with the highest validation NSE
     (mean over seeds for the models that use several) is selected.
  3. Test: every pair is retrained on 2012-2019 and scored on 2020-2021. The
     full test table is reported and the selected window of each model is
     marked. The test scores are never used for the selection.

Reads   data/features.csv
        result/experiment2/experiment2_results.xlsx
Writes  result/experiment3/experiment3_results.xlsx
        result/experiment3/predictions.csv        (test years, selected pairs)
        result/experiment3/plots/*.png

Examples
  python Experiment3.py
  python Experiment3.py --windows 5 10 20
  python Experiment3.py --seeds 42 21 --models ridge rf
"""
import argparse
from pathlib import Path

import pandas as pd

import config
import modelling
import reporting
from Experiment2 import COMBINATIONS, build_inputs
from feature_generation import load_features

DEFAULT_WINDOWS = [3, 5, 10, 20, 30]     # past days per lagged variable
DEFAULT_OUT_DIR = config.RESULT_DIR / "experiment3"
EXPERIMENT2_RESULTS = config.EXPERIMENT2_DIR / "experiment2_results.xlsx"


def load_combinations(path: Path, models) -> dict:
    """Each model's selected lag combination from the Experiment 2 results."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run Experiment2.py first (or pass --exp2-results).")
    table = pd.read_excel(path, sheet_name="Best per model")
    chosen = dict(zip(table["Model"], table["Selected combination"]))
    combinations = {}
    for model in models:
        if model not in chosen:
            raise ValueError(f"{model} is not in the Experiment 2 results")
        if chosen[model] not in COMBINATIONS:
            raise ValueError(f"Unknown combination: {chosen[model]}")
        combinations[model] = chosen[model]
    return combinations


# ===================== 1. VALIDATION =====================
def validate_windows(df, combinations, windows, seeds) -> pd.DataFrame:
    """Score every model with every window on the validation years (test years excluded)."""
    development = df[df.index <= pd.Timestamp(config.VALIDATION_END)]
    all_runs = []
    for window in windows:
        print(f"\n===== validation | window {window} =====")
        for model, combination in combinations.items():
            print(f"\n----- {model} | {combination} | window {window} -----")
            X, y = build_inputs(development, COMBINATIONS[combination], window)
            X_train, X_val, y_train, y_val = modelling.date_split(
                X, y, config.TRAIN_END)
            runs, _ = modelling.compare_models(
                X_train, y_train, X_val, y_val, models=[model], seeds=seeds)
            runs.insert(0, "Combination", combination)
            runs.insert(0, "Window", window)
            all_runs.append(runs)
    return pd.concat(all_runs, ignore_index=True)


# ===================== 2. SELECTION =====================
def choose_windows(val_summary: pd.DataFrame) -> pd.DataFrame:
    """For each model, the window with the highest validation NSE."""
    rows = []
    for model, group in val_summary.groupby("Model", sort=False):
        best = group.loc[group["NSE"].idxmax()]
        rows.append({
            "Model": model,
            "Combination (Experiment 2)": best["Combination"],
            "Selected window": int(best["Window"]),
            "Validation NSE": best["NSE"],
        })
    return pd.DataFrame(rows)


# ===================== 3. TEST =====================
def test_all(df, combinations, windows, seeds):
    """Retrain every model-window pair on 2012-2019 and score it once on the
    test years. Returns the runs, the predictions and the reference forecasts."""
    test_runs, predictions = [], {}
    for window in windows:
        print(f"\n===== test | window {window} =====")
        for model, combination in combinations.items():
            print(f"\n----- {model} | {combination} | window {window} -----")
            X, y = build_inputs(df, COMBINATIONS[combination], window)
            X_dev, X_test, y_dev, y_test = modelling.date_split(
                X, y, config.VALIDATION_END)
            runs, preds = modelling.compare_models(
                X_dev, y_dev, X_test, y_test, models=[model], seeds=seeds)
            runs.insert(0, "Combination", combination)
            runs.insert(0, "Window", window)
            test_runs.append(runs)
            predictions[(model, window)] = preds

    # reference forecasts do not depend on the window, so they are built from
    # the full target series and added once
    target = df[config.TARGET]
    observed = next(iter(predictions.values()))["Observed"]
    development = target[target.index <= pd.Timestamp(config.VALIDATION_END)]
    references = {
        "Reference: seasonal average":
            modelling.seasonal_average(development, observed.index),
        "Reference: yesterday's discharge":
            modelling.yesterday(target, observed.index),
    }
    reference_runs = [{
        "Window": "", "Combination": "", "Model": name, "Seed": "",
        **modelling.get_metrics(observed.values, pred), "Best_params": ""}
        for name, pred in references.items()]
    test_runs.append(pd.DataFrame(reference_runs))
    return pd.concat(test_runs, ignore_index=True), predictions, references


def selected_predictions(predictions, references, selected_window: dict,
                         combinations: dict):
    """Test-year predictions of each model with its selected window."""
    first = next(iter(predictions.values()))
    out = pd.DataFrame({"Observed": first["Observed"]})
    for model, window in selected_window.items():
        preds = predictions[(model, window)]
        for column in preds.columns:
            if column != "Observed" and column.startswith(model):
                out[f"{column} - {combinations[model]}, {window} days"] = preds[column]
    for name, pred in references.items():
        out[name] = pred
    return out


# ===================== RUN =====================
def run(features_path: Path = config.FEATURES_PATH,
        exp2_results: Path = EXPERIMENT2_RESULTS,
        out_dir: Path = DEFAULT_OUT_DIR,
        windows=None,
        models=None,
        seeds=None):
    windows = sorted(windows) if windows else list(DEFAULT_WINDOWS)
    models = list(models) if models else list(modelling.MODELS)
    seeds = list(seeds) if seeds else list(config.NN_SEEDS)

    combinations = load_combinations(exp2_results, models)
    print("Lag combination of each model (from Experiment 2):")
    for model, combination in combinations.items():
        print(f"  {model}: {combination}")
    df = load_features(features_path)
    print(f"windows {windows} | train 2012-{config.TRAIN_END[:4]} | "
          f"validation up to {config.VALIDATION_END} | test after that")

    # validation and selection (test years not used)
    val_runs = validate_windows(df, combinations, windows, seeds)
    val_summary = reporting.summarise(val_runs, by=["Model", "Window"])
    val_summary.insert(2, "Combination",
                       val_summary["Model"].map(combinations))
    choice = choose_windows(val_summary)
    selected = dict(zip(choice["Model"], choice["Selected window"]))

    # test: every model-window pair, selected ones marked
    test_runs, predictions, references = test_all(
        df, combinations, windows, seeds)
    test_summary = reporting.summarise(test_runs, by=["Model", "Window"])
    test_summary.insert(2, "Combination",
                        test_summary["Model"].map(combinations).fillna(""))
    test_summary.insert(3, "Selected", [
        "yes" if selected.get(m) == w else ""
        for m, w in zip(test_summary["Model"], test_summary["Window"])])
    test_selected = pd.concat(
        [test_summary[test_summary["Selected"] == "yes"],
         test_summary[test_summary["Window"] == ""]], ignore_index=True)

    # one table: validation choice next to its test score
    test_scores = test_summary.set_index(["Model", "Window"])
    for metric in ("NSE", "RMSE", "MAE", "KGE", "PBIAS"):
        choice[f"Test {metric}"] = [
            test_scores.loc[(m, w), metric]
            for m, w in zip(choice["Model"], choice["Selected window"])]

    settings = pd.DataFrame({"Setting": [
        "Windows tried (days per lagged variable)", "Same-day inputs",
        "Train years", "Validation years", "Test years",
        "Neural network seeds", "Experiment 2 results used"], "Value": [
        ", ".join(map(str, windows)), ", ".join(config.BASE_FEATURES),
        f"2012 to {config.TRAIN_END[:4]}",
        f"{int(config.TRAIN_END[:4]) + 1} to {config.VALIDATION_END[:4]}",
        f"{int(config.VALIDATION_END[:4]) + 1} to 2021",
        ", ".join(map(str, seeds)), str(exp2_results)]})

    reporting.write_excel(out_dir / "experiment3_results.xlsx", {
        "Best per model": choice,
        "Validation summary": val_summary,
        "Test summary (all)": test_summary,
        "Test (selected)": test_selected,
        "Validation runs": val_runs,
        "Test runs": test_runs,
        "Settings": settings,
    })
    chosen_predictions = selected_predictions(
        predictions, references, selected, combinations)
    chosen_predictions.round(3).to_csv(out_dir / "predictions.csv")

    plot_dir = out_dir / "plots"
    columns = [f"{w} days" for w in windows]
    chosen_cells = {m: f"{w} days" for m, w in selected.items()}

    def score_table(summary):
        table = summary[summary["Window"] != ""].copy()
        table["Window"] = table["Window"].map(lambda w: f"{int(w)} days")
        return table.pivot(index="Model", columns="Window",
                           values="NSE").reindex(index=models, columns=columns)

    val_table = score_table(val_summary)
    test_table = score_table(test_summary)
    reporting.plot_score_heatmap(
        val_table, chosen_cells, plot_dir / "validation_nse_heatmap.png",
        title="Validation NSE by lag window; outlined = selected",
        colorbar_label="Validation NSE (2018-2019)")
    reporting.plot_score_heatmap(
        test_table, chosen_cells, plot_dir / "test_nse_heatmap.png",
        title="Test NSE by lag window; outlined = selected on validation",
        colorbar_label="Test NSE (2020-2021)")
    reporting.plot_predictions(chosen_predictions, plot_dir)

    print("\nValidation NSE (mean over seeds):")
    print(val_table.round(3).to_string())
    print("\nTest NSE (mean over seeds):")
    print(test_table.round(3).to_string())
    print("\nBest lag window for each model:")
    print(choice[["Model", "Combination (Experiment 2)", "Selected window",
                  "Validation NSE", "Test NSE"]].round(3).to_string(index=False))
    print(f"\nResults saved to {out_dir}")
    return choice


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--windows", type=int, nargs="+",
                        help=f"windows to try (default {DEFAULT_WINDOWS})")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help=f"seeds for RF, XGBoost, ANN and LSTM (default {config.NN_SEEDS})")
    parser.add_argument("--models", nargs="+",
                        choices=sorted(modelling.MODEL_ALIASES),
                        help="models to run (default: all six)")
    parser.add_argument("--exp2-results", type=Path, default=EXPERIMENT2_RESULTS,
                        help="Experiment 2 results file to read the combinations from")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="folder for the results (default result/experiment3)")
    args = parser.parse_args(argv)

    models = ([modelling.MODEL_ALIASES[m] for m in args.models]
              if args.models else None)
    run(exp2_results=args.exp2_results, out_dir=args.out_dir,
        windows=args.windows, models=models, seeds=args.seeds)


if __name__ == "__main__":
    main()