# River Streamflow Prediction

Comparison of Random Forest, XGBoost, SVM, Ridge Regression, ANN and LSTM for
daily water discharge of the Kabul River at Nowshera (2012-2021). Weather inputs
come from Peshawar.

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
  Experiment2.py          seven lag combinations, best one per model
  Experiment3.py          lag window length, best one per model
  Experiment4.py          forecasts 3, 5 and 7 days ahead
  pipeline.py             runs the stages above in order
result/                   created when the scripts run
```

`modelling.py` and `reporting.py` do not read or write the project files. The
experiment scripts load the data, choose the settings and call them.

## Data split

| Period | Years | Use |
|---|---|---|
| Train | 2012-2017 | fitting while settings are being chosen |
| Validation | 2018-2019 | choosing settings (Experiments 2 and 3) |
| Test | 2020-2021 | final scores; never used for any choice |

Splits are chronological. After a choice is made, the models are retrained on
2012-2019 and scored once on 2020-2021.

## Experiments

| Experiment | Question | Selection |
|---|---|---|
| 1 | How do the six models do with no lag features (day-of-year and same-day weather only)? | none |
| 2 | Which lag combination suits each model? Seven combinations of discharge, rainfall and maximum temperature, at one window (default 10 days) | highest validation NSE per model |
| 3 | How many past days should each model see? Windows of 3, 5, 10, 20 and 30 days, with each model's combination from Experiment 2 | highest validation NSE per model |
| 4 | How far ahead can each model forecast? Leads of 3, 5 and 7 days, using each model's combination and window from Experiment 3 | none |

In Experiment 4 the inputs are lags that end h days before the target, plus the
day-of-year features. Weather of the target day is not used.

The settings are chosen by the highest validation NSE, with no tolerance. Every
model gets the same tuning effort (TimeSeriesSplit with 5 folds, NSE scoring;
8 settings for ANN and LSTM). RF, XGBoost, ANN and LSTM are run with five seeds
(42, 21, 256, 512, 1024) and the scores are averaged; SVM and Ridge are
deterministic.

## Metrics

NSE, RMSE, MAE, KGE, PBIAS (positive means underestimation), and RMSE and PBIAS
on the high-flow days (the top 5% of observed discharge). Each table also
carries two references: the seasonal average (mean of the training years for each
calendar day) and persistence (discharge h days earlier, h being the lead time).

## Running

Run from the `src` folder (in PyCharm, mark `src` as Sources Root).

```
python pipeline.py                       # preprocess, eda, features, feature_plots
python pipeline.py --steps experiment1   # trains models, a few minutes
python pipeline.py --steps experiment2   # several hours
python pipeline.py --steps experiment3   # several hours, needs Experiment 2
python pipeline.py --steps experiment4   # needs Experiment 3
python pipeline.py --steps all

python Experiment1.py                    # defaults from config.py
python Experiment1.py --seeds 42 21 --models ridge rf

python Experiment2.py                    # window 10, results in result/experiment2
python Experiment2.py --window 3         # same experiment at another window

python Experiment3.py                    # windows 3 5 10 20 30
python Experiment3.py --windows 5 10 20
python Experiment3.py --seeds 42 21 --models ridge rf

python Experiment4.py
```

Models for `--models`: rf, xgb, svm, ridge, ann, lstm.

The experiments depend on each other: 2 reads nothing from the others, 3 reads
`result/experiment2/experiment2_results.xlsx`, and 4 reads the "Best per model"
sheet of `result/experiment3/experiment3_results.xlsx`. Run them in order.

## Output

Each experiment writes to `result/experimentN/`:

- `experimentN_results.xlsx` with the validation and test tables, the individual
  runs and the settings used
- `predictions.csv` with the test-year predictions of the selected models and the
  two references (Experiments 3 and 4)
- `plots/` with the figures

In Experiments 2 and 3 the test table lists every combination or window, with
the selected one marked. Report the selected row as the model's result; the other
rows are descriptive. The prediction files contain the mean over seeds for RF,
XGBoost, ANN and LSTM, so scores computed from them are slightly higher than the
per-seed means in the results workbooks.

## Setup

```
pip install -r requirements.txt
```
