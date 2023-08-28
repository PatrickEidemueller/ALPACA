from typing import Union

import tqdm

from alpaca.system.utils.printing import to_string
from alpaca.system._session import ActiveLearningSession
from alpaca.system.files import FileUtils
from alpaca.system.frameworks._conversion import convert_train_pool_set, convert_dataset

from alpaca.system.interfaces.dataset import Dataset, TrainPoolSet
from alpaca.system.interfaces.model import ModelInterface
from alpaca.system.interfaces.selector import SelectorInterface

from alpaca.system.utils.stopwatch import StopWatch

from alpaca.system.reports import IterationReport, PerformanceReport


def evaluate_model(
    model: ModelInterface, train_set: Dataset, pool_set: Dataset, test_set: Dataset
) -> PerformanceReport:
    with StopWatch() as stopwatch:
        train_rmse = model.RMSE(train_set)
        pool_rmse = model.RMSE(pool_set)
        test_rmse = model.RMSE(test_set)
        return PerformanceReport(
            train_size=len(train_set),
            pool_size=len(pool_set),
            test_size=len(test_set),
            train_rmse=train_rmse,
            pool_rmse=pool_rmse,
            test_rmse=test_rmse,
            timestamp_start=stopwatch.start,
            duration=stopwatch.elapsed(),
        )


def do_one_active_learning_iteration(
    batch_size: int,
    iteration: int,
    model: ModelInterface,
    tp_set: TrainPoolSet,
    test_set: Dataset,
    selector: SelectorInterface
) -> IterationReport:
    with StopWatch(f"Iteration") as stopwatch_it:
        fit_artifact = model.fit(tp_set.train)
        evaluation_artifact = evaluate_model(model, tp_set.train, tp_set.pool, test_set)
        selection_artifact = selector.select_batch(model, tp_set, batch_size)
        return IterationReport(
            iteration=iteration,
            model_fit=fit_artifact,
            performance=evaluation_artifact,
            selection=selection_artifact,
            timestamp_start=stopwatch_it.start,
            duration=stopwatch_it.elapsed(),
        )


def do_active_learning(
    session: ActiveLearningSession,
    batch_size: Union[int, float],
    iterations: int = None,
    save_model: bool = False,
) -> None:
    if isinstance(batch_size, float):
        batch_size = int(len(session.pool_set) * batch_size)
    if iterations is None:
        iterations = int(len(session.pool_set) / batch_size)

    # Convert dataset (does not change dataset when it already has the required type)
    convert_train_pool_set(
        session._train_pool_set, session._model.RequiredDataLoaderType
    )
    convert_dataset(session._test_set, session._model.RequiredDataLoaderType)

    for it in tqdm.tqdm(range(iterations)):
        FileUtils.save_session(session)
        iteration_report: IterationReport = do_one_active_learning_iteration(
            batch_size,
            session.iteration,
            session._model,
            session._train_pool_set,
            session.test_set,
            session.selector,
        )
        FileUtils.save_iteration_report(session.name, iteration_report)
        if save_model:
            FileUtils.save_session_model(session)
        # Reset model
        session.next_iteration()
    FileUtils.save_session(session)
