# River Streamflow Prediction

Comparison of Random Forest, XGBoost, SVM, Ridge Regression, ANN and LSTM for
daily water discharge of the Kabul River at Nowshera (2012-2021).

## Layout

```
data/flood_data.xlsx      raw data (input)
src/
  config.py               paths and settings shared by all scripts
  Preprocessing.py        raw Excel -> data/flood_clean_data.csv
  EDA.py                  plots of the clean data -> result/eda_plots/
  feature_generation.py   day-of-year features -> data/features.csv; make_lags()
  feature_plots.py        plot of the day-of-year features
  modelling.py            the six models, metrics, compare_models()
  reporting.py            result tables and figures
  Experiment1.py          six models without lag features
  pipeline.py             runs the stages above in order
result/                   created when the scripts run
```

`modelling.py` and `reporting.py` do not read or write the project files. The
experiment scripts load the data, choose the settings and call them.

## Running

Run from the `src` folder (in PyCharm, mark `src` as Sources Root).

```
python pipeline.py                       # preprocess, eda, features, feature_plots
python pipeline.py --steps experiment1   # trains models, a few minutes
python pipeline.py --steps all

python Experiment1.py                    # defaults from config.py
python Experiment1.py --seeds 42 21 --models ridge rf
```

Models for `--models`: rf, xgb, svm, ridge, ann, lstm.

## Setup

```
pip install -r requirements.txt
```
