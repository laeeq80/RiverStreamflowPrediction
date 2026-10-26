"""
Preprocessing
Clean the raw Kabul River daily data (Nowshera, 2012-2021).

Reads  data/flood_data.xlsx
Writes data/flood_clean_data.csv

Cleaning steps, each confirmed against the raw file:
  1. Maximum/Minimum_Temperature: some cells are numbers followed by a
     non-breaking space; the whitespace is stripped.
  2. Rainfall: 271 entries are written as text (trace / Trace / TRACE / Tr /
     TR). A trace amount is set to 0.05 mm.
  3. Average_Temperature: in the sheet this is =AVERAGE(max, min). Excel skips
     cells stored as text, so on the 15 days where a temperature was text the
     "average" is just the other temperature. It is recomputed as (max+min)/2.
Any other unreadable value raises an error instead of being repaired.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "flood_data.xlsx"
CLEAN_PATH = PROJECT_ROOT / "data" / "flood_clean_data.csv"

TRACE_LABELS = {"trace", "tr"}
TRACE_MM = 0.05
TEMP_COLS = ["Maximum_Temperature", "Minimum_Temperature"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TEMP_COLS:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip())

    sheet_avg = df["Average_Temperature"]
    df["Average_Temperature"] = ((df["Maximum_Temperature"]
                                  + df["Minimum_Temperature"]) / 2).round(2)
    changed = (df["Average_Temperature"] - sheet_avg).abs() > 0.01
    print(f"Average_Temperature recomputed, rows that changed: {changed.sum()}")

    rain = df["Rainfall"].astype(str).str.strip().str.lower()
    is_trace = rain.isin(TRACE_LABELS)
    df["Rainfall"] = pd.to_numeric(rain.mask(is_trace, str(TRACE_MM)))
    print(f"Trace rainfall entries set to {TRACE_MM} mm: {is_trace.sum()}")

    return df


def check(df: pd.DataFrame) -> None:
    """Stop if the series has gaps, duplicates or missing values."""
    if df.isna().any().any():
        raise ValueError("Missing values found:\n" + str(df.isna().sum()))

    days = df["Date"].sort_values()
    full_range = pd.date_range(days.min(), days.max(), freq="D")
    if not days.is_unique or len(full_range) != len(days):
        raise ValueError("Dates are duplicated or the daily record has gaps")


def main() -> None:
    df = pd.read_excel(RAW_PATH)
    df = clean(df)
    check(df)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"{len(df)} rows saved to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
