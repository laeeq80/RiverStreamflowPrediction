"""Paths and settings shared by every script in the project.

Change a value here and every stage picks it up. Scripts can still override
most of these through arguments of their run() function.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- data files
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "flood_data.xlsx"
CLEAN_PATH = DATA_DIR / "flood_clean_data.csv"
FEATURES_PATH = DATA_DIR / "features.csv"

# ---- result folders
RESULT_DIR = PROJECT_ROOT / "result"
EDA_PLOT_DIR = RESULT_DIR / "eda_plots"
FEATURE_PLOT_DIR = RESULT_DIR / "feature_plots"
EXPERIMENT1_DIR = RESULT_DIR / "experiment1"

# ---- columns
TARGET = "Water_Discharge"
# same-day model inputs; lag columns are added by the experiments
BASE_FEATURES = [
    "Maximum_Temperature", "Minimum_Temperature", "Rainfall",
    "DayOfYear_sin", "DayOfYear_cos",
]
DAYS_PER_YEAR = 365.25       # period of the day-of-year encoding

# ---- modelling
TRAIN_FRACTION = 0.8         # first 80% of days train, last 20% test
HIGH_FLOW_QUANTILE = 0.95    # "high-flow days" = top 5% of observed discharge
CV_SPLITS = 5                # time-ordered folds used for tuning
RANDOM_STATE = 42            # Random Forest and XGBoost
NN_SEEDS = [42, 21, 256, 512, 1024]   # ANN and LSTM are run once per seed

# ---- figures
FIGURE_DPI = 300
