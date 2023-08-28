import argparse
from os.path import exists, join

from alpaca.visualization.plot_selected_2d import plot_selected_per_iteration
import alpaca.runner as alp 

from alpaca.system.files import BASE_DIR

SYN_DATASPLIT_NAME = "_synthetic_2d_vis2d"

def _prepare_data() -> None:
    if not exists(join(BASE_DIR, "sessions", ".data_states", SYN_DATASPLIT_NAME)):
        alp.split("_synthetic_2d_vis2d", "synthetic_2dims", 0.01, 0.69, 0.3, 42)

def main():
    parser = argparse.ArgumentParser(
        description="Runs the experiment with the selected model on 2d synthetic data. Plots the datapoints selected at each iteration")
    parser.add_argument(
        "selector",
        type=str,
        nargs="?",
        default="*",
        help="The name of the selector to use. '*' to run with all selectors. Default: '*'",
    )
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default="*",
        help="Model type to use in the experiment. '*' to use all compatible models. Default: '*'",
    )
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Start a new experiment, any existing savefiles will be deleted",
    )
    args = parser.parse_args()

    _prepare_data()
    non_empty = False
    for experiment_name in alp.run(SYN_DATASPLIT_NAME, args.selector, args.model, args.new):
        plot_selected_per_iteration(experiment_name)
        non_empty = True
    assert non_empty


if __name__=="__main__":
    main()