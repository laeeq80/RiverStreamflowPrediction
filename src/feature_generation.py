"""Feature generation for the Kabul River discharge models.

Run as a script, it adds the fixed features and saves them:
  data/flood_clean_data.csv  ->  data/features.csv

The experiment scripts import make_lags() to build lag columns with the
settings of each experiment, so lag columns are not saved to a file, and
load_features() to read the saved table.

Fixed features
  DayOfYear_sin, DayOfYear_cos : position in the year on a circle (period
      365.25 days), so 31 December and 1 January are close to each other.
Date is kept as the index and is never used as a model input.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config


def load_clean(path: Path = config.CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df.sort_index()


def load_features(path: Path = config.FEATURES_PATH) -> pd.DataFrame:
    """Read the saved feature table (Date is the index)."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    missing = [c for c in config.BASE_FEATURES + [config.TARGET]
               if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns {missing}; "
                       "run the features step again")
    return df


def check_daily(df: pd.DataFrame) -> None:
    """Lags are shifts by rows, so every row must be exactly one day apart."""
    steps = df.index.to_series().diff().dropna()
    if not (steps == pd.Timedelta(days=1)).all():
        raise ValueError("The index is not a continuous daily series")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    angle = 2 * np.pi * df.index.day_of_year / config.DAYS_PER_YEAR
    df["DayOfYear_sin"] = np.sin(angle)
    df["DayOfYear_cos"] = np.cos(angle)
    return df


def make_lags(df: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    """Add lag columns, e.g. lags={"Water_Discharge": 10, "Rainfall": 5}.

    Column <name>_lag_k holds the value of <name> k days earlier. The first
    max(lags.values()) rows have no history and are dropped.
    """
    check_daily(df)
    missing = [c for c in lags if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if any(n < 1 for n in lags.values()):
        raise ValueError("Each lag count must be at least 1")

    out = df.copy()
    for col, n in lags.items():
        for k in range(1, n + 1):
            out[f"{col}_lag_{k}"] = df[col].shift(k)
    return out.iloc[max(lags.values()):]


def run(clean_path: Path = config.CLEAN_PATH,
        features_path: Path = config.FEATURES_PATH) -> pd.DataFrame:
    df = add_calendar_features(load_clean(clean_path))
    check_daily(df)
    if df.isna().any().any():
        raise ValueError("Missing values in the feature table")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_path)
    print(f"{len(df)} rows, columns: {list(df.columns)}")
    print(f"saved to {features_path}")
    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
