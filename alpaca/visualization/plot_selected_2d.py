import numpy as np
from matplotlib import pyplot as plt

from alpaca.system.interfaces.dataset import Dataset
from alpaca.system.factory import ExperimentState, FactoryUtils
from alpaca.system.files import FileUtils, BASE_DIR
from alpaca.system.utils.filesystem import fjoin
from alpaca.system.reports import IterationReport
from alpaca.system.frameworks.numpy_classes import NumpyDataLoader


TRAIN_COLOR = "lime"
POOL_COLOR = "lightsteelblue"
SELECT_COLOR = "red"

def _get_outpath(session_name: str, iteration: int) -> str:
    return fjoin(BASE_DIR, "reports", "figures_vis2d", session_name, f"iteration{iteration}.png")

def _scatter():
    pass

def _plot_iteration(experiment_state: ExperimentState, iteration_report: IterationReport, outpath: str) -> None:
    tp_data, _ = FactoryUtils.create_data_state(experiment_state.data)
    assert isinstance(tp_data.train.data_loader, NumpyDataLoader)
    data_loader: NumpyDataLoader = tp_data.data_loader
    assert data_loader.x_dims == 2

    training_x, training_y = data_loader.load_batch(experiment_state.data.train_set_ids)
    pool_x, pool_y = data_loader.load_batch(experiment_state.data.pool_set_ids)
    select_x, select_y = data_loader.load_batch(iteration_report.selection.selected_ids)

    plt.scatter(x=training_x[:,0], y=training_x[:,1], c=TRAIN_COLOR, s=3)
    plt.scatter(x=pool_x[:,0], y=pool_x[:,1], c=POOL_COLOR, s=2)
    plt.scatter(x=select_x[:,0], y=select_x[:,1], c=SELECT_COLOR, s=3)
    plt.savefig(outpath)
    plt.clf()

def plot_selected_per_iteration(session_name: str) -> None:
    states: list[ExperimentState] = FileUtils.load_all_experiment_states(session_name)[:-1]
    reports: list[IterationReport] = FileUtils.load_all_iteration_reports(session_name)
    assert len(reports) == len(states)

    for state, report in zip(states, reports):
        _plot_iteration(state, report, _get_outpath(session_name, report.iteration))
    plt.close()
