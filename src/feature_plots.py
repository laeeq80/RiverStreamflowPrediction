"""Figures that describe the generated features.

Reads  data/features.csv
Writes result/feature_plots/day_of_year_encoding.png
"""
import calendar
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features.csv"
PLOT_DIR = PROJECT_ROOT / "result" / "feature_plots"
DPI = 300

# first day of each month in a non-leap year
MONTH_START = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


def day_of_year_encoding(df: pd.DataFrame) -> None:
    doy = df.index.day_of_year
    by_day = (df.assign(day_of_year=doy)
                .groupby("day_of_year")
                .agg(sin=("DayOfYear_sin", "first"),
                     cos=("DayOfYear_cos", "first"),
                     discharge=("Water_Discharge", "mean")))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})

    # left: the two features over the year
    ax1.plot(by_day.index, by_day["sin"], color="tab:blue", label="DayOfYear_sin")
    ax1.plot(by_day.index, by_day["cos"], color="tab:red", label="DayOfYear_cos")
    ax1.set_xticks(MONTH_START)
    ax1.set_xticklabels(calendar.month_abbr[1:13])
    ax1.set_xlim(1, 366)
    ax1.set_xlabel("Day of year (month starts)")
    ax1.set_ylabel("Feature value")
    ax1.set_title("Day-of-year features")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="lower left")

    # right: the same two values as a point on a circle, coloured by the
    # mean discharge recorded on that day of the year
    sc = ax2.scatter(by_day["cos"], by_day["sin"], c=by_day["discharge"],
                     cmap="viridis", s=16)
    for day, name in {1: "Jan", 91: "Apr", 182: "Jul", 274: "Oct"}.items():
        row = by_day.loc[day]
        ax2.annotate(name, (row["cos"], row["sin"]),
                     xytext=(6, 6), textcoords="offset points")
    ax2.set_aspect("equal")
    ax2.set_xlabel("DayOfYear_cos")
    ax2.set_ylabel("DayOfYear_sin")
    ax2.set_title("Position on the yearly circle")
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Mean discharge on that day (m³/s)")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "day_of_year_encoding.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved day_of_year_encoding.png")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"], index_col="Date")
    day_of_year_encoding(df)


if __name__ == "__main__":
    main()
