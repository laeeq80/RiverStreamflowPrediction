"""Run the project stages in order.

  preprocess     Preprocessing.py       raw Excel -> data/flood_clean_data.csv
  eda            EDA.py                 plots of the clean data
  features       feature_generation.py  data/features.csv
  feature_plots  feature_plots.py       plot of the day-of-year features
  experiment1    Experiment1.py         six models without lag features

With no arguments the first four stages run. Experiment 1 trains models and
takes a few minutes, so it only runs when asked for.

Examples
  python pipeline.py
  python pipeline.py --steps eda features
  python pipeline.py --steps experiment1
  python pipeline.py --steps all
"""
import argparse
import importlib

# stage name -> module with a run() function, in the order they run
STEPS = {
    "preprocess": "Preprocessing",
    "eda": "EDA",
    "features": "feature_generation",
    "feature_plots": "feature_plots",
    "experiment1": "Experiment1",
}
DEFAULT_STEPS = ["preprocess", "eda", "features", "feature_plots"]


def run_pipeline(steps=None) -> None:
    chosen = steps or DEFAULT_STEPS
    for name, module_name in STEPS.items():      # always in pipeline order
        if name in chosen:
            print(f"\n=== {name} ===")
            # imported here so that, for example, TensorFlow is only loaded
            # when an experiment is actually run
            importlib.import_module(module_name).run()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", nargs="+", choices=list(STEPS) + ["all"],
                        help="stages to run (default: preprocess, eda, "
                             "features, feature_plots)")
    args = parser.parse_args(argv)

    steps = args.steps
    if steps and "all" in steps:
        steps = list(STEPS)
    run_pipeline(steps)


if __name__ == "__main__":
    main()
