from copy import deepcopy

from alpaca.system.factory import (
    ExperimentState,
    DataState,
    ModelState,
    SelectorState,
)

from alpaca.system.interfaces.dataset import Dataset, TrainPoolSet
from alpaca.system.interfaces.model import ModelInterface
from alpaca.system.interfaces.selector import SelectorInterface

from alpaca.system.utils.seeding import set_seed
from alpaca.system.utils.printing import add_indentation
from alpaca.system.factory import FactoryUtils


class ActiveLearningSession:
    """
    Contains the current state of an active learning experiment.
    """
    def __init__(self, name: str, experiment_state: ExperimentState, iteration: int):
        """
        @param name : Name of the session. The name is used to determine the folder in which
            the session will be saved.
        @param experiment_state : The current state of the datasets, model and selector
        @param iteration : The current iteration. Is used to determine the file names under which
            the different states / results at different iterations are saved in the session folder.
        """
        self._name: str = name
        self._iteration: int = iteration
        self._experiment_state: ExperimentState = experiment_state
        self._model: ModelInterface = None
        self._selector: SelectorInterface = None
        self._train_pool_set: TrainPoolSet = None
        self._test_set: Dataset = None
        # Creates _model, _selector, _train_pool_set, _test_set
        self.set_experiment_state(experiment_state)

    @property
    def name(self) -> str:
        return self._name

    @property
    def iteration(self) -> str:
        return self._iteration

    @property
    def experiment_state(self) -> ExperimentState:
        return self._experiment_state

    @property
    def train_set(self) -> Dataset:
        return self._train_pool_set.train

    @property
    def pool_set(self) -> Dataset:
        return self._train_pool_set.pool

    @property
    def test_set(self) -> Dataset:
        return self._test_set

    @property
    def model(self) -> ModelInterface:
        return self._model

    @property
    def selector(self) -> SelectorInterface:
        return self._selector

    def set_data_state(self, data_state: DataState) -> None:
        self._experiment_state.data = deepcopy(data_state)
        self._train_pool_set, self._test_set = FactoryUtils.create_data_state(
            self._experiment_state.data
        )

    def set_model_state(self, model_state: ModelState) -> None:
        self._experiment_state.model = model_state
        self._model = FactoryUtils.create_model_state(self._experiment_state.model)

    def set_selector_state(self, selector_state: SelectorState) -> None:
        self._experiment_state.selector = selector_state
        self._selector = FactoryUtils.create_selector_state(
            self._experiment_state.selector
        )

    def set_experiment_state(self, experiment_state: ExperimentState) -> None:
        set_seed(self._experiment_state.seed)
        self.set_data_state(experiment_state.data)
        self.set_model_state(experiment_state.model)
        self.set_selector_state(experiment_state.selector)

    def next_iteration(self) -> None:
        """
        Updates the iteration count, sets the seed given in the experiment state.
        Updates the ids in the experiment state to the current ids of train, pool and test sets.
        """
        self._iteration += 1
        set_seed(self._experiment_state.seed)
        self._experiment_state.model.parameter_file = None
        self._experiment_state.data.train_set_ids = self.train_set.ids
        self._experiment_state.data.pool_set_ids = self.pool_set.ids
        self._experiment_state.data.test_set_ids = self.test_set.ids
        self._model = FactoryUtils.create_model_state(self._experiment_state.model)
