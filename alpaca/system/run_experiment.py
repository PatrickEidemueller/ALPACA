from alpaca.system.files import FileUtils, SessionException
from alpaca.system.factory import (
    ExperimentState,
    DataState,
    ModelState,
    SelectorState,
)
from alpaca.system._active_learning import do_active_learning


def run_experiment(
    name: str,
    experiment_state: ExperimentState,
    batch_size: int,
    max_iteration: int = None,
    new_experiment: bool = False,
    copy_data_split_from: str = None,
    save_model: bool = False,
):
    """
    Runs the experiment given by the experiment state until max_iteration is reached.
    Any results are stored in the session given by <experiment_name>. If the session already exists,
    the state of the existing session is loaded and the experiment is run from there. Otherwise a new
    session is created and the experiment starts at iteration 1.

    @param name : Name of the session in which the experiment should be saved. If this
        session already exists and "new_experiment" is False the existing state is loaded.
    @param experiment_state : The experiment that should be run.
    @param batch_size : The number of pool elements that are labeled at each iteration.
    @param max_iteration : After finishing iteration <max_iteration> the experiment is finished.
        If a new experiment is created this equivalent to the number of iterations run.
        If an existing state is loaded then the number of iterations is
        <max_iteration> - <current_iteration> + 1
    @param new_experiment : Always start the experiment at iteration 1. This deletes all pre-existing
        files of the session with the experiment's name.
    @param copy_data_split_from : Name of a saved (FileUtils.save_data_state()) data_state.
        The data_state must have the same number of indices as the data_state of the experiment.
    @param save_models : Whether to save the trained model parameters of all iterations
    """
    if new_experiment:
        FileUtils.delete_session(name)
    try:
        session = FileUtils.load_session(name)
        # If loading of previous state was successful
        print(
            f"run_experiment {name}: "
            f"Found experiment state at iteration {session.iteration}."
        )
        if experiment_state is not None:
            print(
                f"run_experiment {name}: Argument 'experiment_state' ignored, "
                "as an existing experiment state was loaded."
            )
        if copy_data_split_from is not None:
            print(
                f"run_experiment {name}: Argument 'use_ids_from' ignored, "
                "as an existing experiment state was loaded."
            )
    except SessionException:
        # If no previous experiment state could be loaded
        if experiment_state is None:
            raise RuntimeError(
                f"run_experiment {name}: "
                "Argument 'experiment_state' is None but no existing experiment state could be loaded."
            )
        if copy_data_split_from is not None:
            data_state = FileUtils.load_data_state(copy_data_split_from)
            experiment_state.data.copy_ids(data_state)
        session = FileUtils.new_session(name, experiment_state)

    if max_iteration is None:
        max_iteration = int(len(session.pool_set) / batch_size)
    iterations = (max_iteration - session.iteration) + 1
    if iterations <= 0:
        print(f"run_experiment {name}: Nothing to do.")
        return
    print(
        f"run_experiment {name}: Run for {iterations} iterations "
        f"({session.iteration}, ..., {max_iteration})."
    )
    do_active_learning(
        session=session,
        iterations=iterations,
        batch_size=batch_size,
        save_model=save_model,
    )


def change_experiment(
    experiment_name: str,
    data_state: DataState = None,
    model_state: ModelState = None,
    selector_state: SelectorState = None,
) -> None:
    try:
        session = FileUtils.load_session(experiment_name)
    except SessionException:
        RuntimeError(f"change_experiment {experiment_name}: Experiment does not exist!")
    if data_state is not None:
        session.set_data_state(data_state)
        print(f"change_experiment {experiment_name}: Changed data state.")
    if model_state is not None:
        session.set_model_state(model_state)
        print(f"change_experiment {experiment_name}: Changed model state.")
    if selector_state is not None:
        session.set_selector_state(selector_state)
        print(f"change_experiment {experiment_name}: Changed selector state.")
    FileUtils.save_session(session)
