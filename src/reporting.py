"""Tables and figures for model comparisons.

Shared by the experiment scripts. Takes the output of
modelling.compare_models() and writes it to a folder.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import config
from modelling import METRICS, get_metrics

PREDICTION_PLOT_DPI = 150
# one-hue light-to-dark ramp for magnitudes (heatmaps)
BLUE_RAMP = ["#cde2fb", "#3987e5", "#0d366b"]
INK = "#0b0b0b"


def summarise(runs: pd.DataFrame, by="Model") -> pd.DataFrame:
    """One row per model (or per `by` group): mean of each metric and, for
    several runs, its std."""
    grouped = runs.groupby(by, sort=False)
    summary = grouped[METRICS].mean().join(
        grouped[METRICS].std().add_suffix("_std"))
    summary.insert(0, "Runs", grouped.size())
    return summary.reset_index()


def file_name(text: str) -> str:
    """Safe file name from a title, e.g. 'ANN (mean of seeds)' -> 'ann_mean_of_seeds'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


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
        fig.savefig(plot_dir / (file_name(name) + ".png"),
                    dpi=PREDICTION_PLOT_DPI,
                    bbox_inches="tight")
        plt.close(fig)


def write_excel(path: Path, sheets: dict) -> None:
    """Write {sheet name: dataframe} to one Excel file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for name, frame in sheets.items():
            frame.round(4).to_excel(writer, sheet_name=name, index=False)


def plot_score_heatmap(table: pd.DataFrame, chosen: dict, path: Path,
                       title: str, colorbar_label: str) -> None:
    """Heatmap of one score: rows are models, columns are settings.

    table   DataFrame (models x settings) of the score
    chosen  {model: setting} cells to outline (the setting picked per model)
    """
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(figsize=(1.3 * n_cols + 2.5, 0.75 * n_rows + 2.6))
    values = table.values.astype(float)
    low, high = values.min(), values.max()
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=low, vmax=high)

    for i in range(n_rows):
        for j in range(n_cols):
            dark = (values[i, j] - low) / (high - low + 1e-12) > 0.55
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color="white" if dark else INK)
    for i, model in enumerate(table.index):
        if model in chosen:
            j = list(table.columns).index(chosen[model])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=2.5))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c.replace(" + ", "\n+ ") for c in table.columns],
                       fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(table.index)
    ax.set_title(title, fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
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
