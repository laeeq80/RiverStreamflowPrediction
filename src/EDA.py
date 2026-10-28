"""Exploratory plots for the cleaned Kabul River data (Nowshera, 2012-2021).

Reads  data/flood_clean_data.csv
Writes result/eda_plots/*.png and result/data_summary.csv

The data is only read here, never changed. Month is taken from the Date
column when a plot needs it and is not stored in the dataframe.
"""
import calendar
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

MONTHS = range(1, 13)
MONTH_LABELS = calendar.month_abbr[1:13]

LABELS = {
    "Maximum_Temperature": "Maximum temperature (°C)",
    "Rainfall": "Rainfall (mm)",
    "Water_Discharge": "Water discharge (m³/s)",
}


def save(fig: plt.Figure, plot_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(plot_dir / name, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def month_axis(ax: plt.Axes) -> None:
    ax.set_xticks(list(MONTHS))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")


def temperature_vs_discharge(df: pd.DataFrame, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x=df["Maximum_Temperature"], y=df["Water_Discharge"], order=4,
                scatter_kws={"s": 10, "alpha": 0.4, "color": "tab:blue"},
                line_kws={"color": "tab:red"}, ax=ax)
    ax.set_xlabel(LABELS["Maximum_Temperature"])
    ax.set_ylabel(LABELS["Water_Discharge"])
    ax.set_title("Maximum temperature and water discharge (4th-order fit)")
    save(fig, plot_dir, "1.max_temperature_vs_discharge.png")


def rainfall_vs_discharge(df: pd.DataFrame, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x=df["Rainfall"], y=df["Water_Discharge"],
                scatter_kws={"s": 10, "alpha": 0.4, "color": "tab:blue"},
                line_kws={"color": "tab:red"}, ax=ax)
    ax.set_xlabel(LABELS["Rainfall"])
    ax.set_ylabel(LABELS["Water_Discharge"])
    ax.set_title("Rainfall and water discharge (linear fit)")
    save(fig, plot_dir, "2.rainfall_vs_discharge.png")


def monthly_scatter(df: pd.DataFrame, month: pd.Series, column: str,
                    title: str, plot_dir: Path, name: str) -> None:
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.12, 0.12, len(df))  # keeps the points from stacking
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(month + jitter, df[column], s=14, alpha=0.5, edgecolors="none")
    month_axis(ax)
    ax.set_ylabel(LABELS[column])
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    save(fig, plot_dir, name)


def monthly_rainfall_summary(df: pd.DataFrame, month: pd.Series,
                             plot_dir: Path) -> None:
    mean_rain = df.groupby(month)["Rainfall"].mean().reindex(MONTHS, fill_value=0)
    rainy_days = (df["Rainfall"] > 0).groupby(month).sum().reindex(MONTHS, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(MONTHS, mean_rain, color="steelblue", edgecolor="white")
    for m, v in mean_rain.items():
        ax1.text(m, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    month_axis(ax1)
    ax1.set_ylabel("Mean daily rainfall (mm)")
    ax1.set_title("Mean daily rainfall per month")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    ax2.bar(MONTHS, rainy_days, color="coral", edgecolor="white")
    for m, v in rainy_days.items():
        ax2.text(m, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    month_axis(ax2)
    ax2.set_ylabel("Number of days")
    ax2.set_title("Days with any rainfall (trace included), 2012-2021")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    save(fig, plot_dir, "5.monthly_rainfall_summary_10_Years.png")


def monthly_max_temperature(df: pd.DataFrame, month: pd.Series,
                            plot_dir: Path) -> None:
    groups = [df.loc[month == m, "Maximum_Temperature"] for m in MONTHS]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(groups, positions=list(MONTHS), widths=0.6)
    month_axis(ax)
    ax.set_ylabel(LABELS["Maximum_Temperature"])
    ax.set_title("Maximum temperature by month, 2012-2021")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    save(fig, plot_dir, "6.monthly_max_temperature_summary_10_years.png")


def run(clean_path: Path = config.CLEAN_PATH,
        plot_dir: Path = config.EDA_PLOT_DIR) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(clean_path, parse_dates=["Date"])
    month = df["Date"].dt.month

    summary = df.drop(columns="Date").describe().T
    summary.to_csv(plot_dir.parent / "data_summary.csv")

    temperature_vs_discharge(df, plot_dir)
    rainfall_vs_discharge(df, plot_dir)
    monthly_scatter(df, month, "Water_Discharge",
                    "Monthly distribution of water discharge, Nowshera, 2012-2021",
                    plot_dir, "3.monthly_discharge.png")
    monthly_scatter(df, month, "Rainfall",
                    "Monthly distribution of daily rainfall, 2012-2021",
                    plot_dir, "4.monthly_rainfall.png")
    monthly_rainfall_summary(df, month, plot_dir)
    monthly_max_temperature(df, month, plot_dir)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
