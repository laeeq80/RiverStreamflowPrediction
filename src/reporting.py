"""Tables and figures for model comparisons.

Shared by the experiment scripts. Takes the output of
modelling.compare_models() and writes it to a folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import pandas as pd

from modelling import METRICS, get_metrics

PREDICTION_PLOT_DPI = 150


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean of each metric and, for several runs, its std."""
    by_model = runs.groupby("Model", sort=False)
    summary = by_model[METRICS].mean().join(
        by_model[METRICS].std().add_suffix("_std"))
    summary.insert(0, "Runs", by_model.size())
    return summary.reset_index()


def plot_predictions(predictions: pd.DataFrame, plot_dir: Path) -> None:
    """One observed-versus-predicted figure per model (not the references)."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    obs = predictions["Observed"]
    for name in predictions.columns:
        if name == "Observed" or name.startswith("Reference"):
            continue
        pred = predictions[name].values
        m = get_metrics(obs.values, pred)

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
        fig.savefig(plot_dir / file, dpi=PREDICTION_PLOT_DPI,
                    bbox_inches="tight")
        plt.close(fig)


def save_results(out_dir: Path, runs: pd.DataFrame, summary: pd.DataFrame,
                 predictions: pd.DataFrame, excel_name: str) -> None:
    """Excel file (summary + every run), predictions csv and the plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_dir / excel_name) as writer:
        summary.round(4).to_excel(writer, sheet_name="Summary", index=False)
        runs.round(4).to_excel(writer, sheet_name="All runs", index=False)
    predictions.round(3).to_csv(out_dir / "predictions.csv")
    plot_predictions(predictions, out_dir / "plots")


def print_summary(summary: pd.DataFrame) -> None:
    columns = ["Model", "Runs", "NSE", "RMSE", "MAE", "KGE", "PBIAS",
               "High_RMSE", "High_PBIAS"]
    print("\n" + summary[columns].round(3).to_string(index=False))
