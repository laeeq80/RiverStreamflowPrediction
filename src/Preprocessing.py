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
Any other unreadable value raises an error instead of being repaired.
"""
from pathlib import Path

import pandas as pd

import config

TRACE_LABELS = {"trace", "tr"}
TRACE_MM = 0.05
TEMP_COLS = ["Maximum_Temperature", "Minimum_Temperature"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TEMP_COLS:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip())

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


def run(raw_path: Path = config.RAW_PATH,
        clean_path: Path = config.CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_excel(raw_path)
    df = clean(df)
    check(df)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    print(f"{len(df)} rows saved to {clean_path}")
    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
