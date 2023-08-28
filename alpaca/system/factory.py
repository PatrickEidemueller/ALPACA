"""
An active learning experiment (at some iteration) can be described by
    1. Its DataState
        a) Which dataset was used? 
        b) Which points are in train, pool and test set?
    2. Its ModelState
        a) Which model was used?
        b) Are there any pretrained parameters?
    3. Its SelectorState
        a) Which selector was used? 

The three Factory subclasses DatasetFactory, ModelFactory and SelectorFactory are used
to created datasets, models and selectors from string keys that can be saved to disk.

To use the factory with your own Dataset, Model or Selector implement a function that
takes only kwargs that are also json serializable (bool, int, float, str, list, dict...)
and returns the respective object. Note that you need to derive your own classes from
DataLoaderInterface, ModelInterface or SelectorInterface (see pipeline.interfaces).
In pipeline.frameworks there are already a good number of useful Base classes that
you can (and most likely should) reuse.  

Register the function in the factory under a key of your choice.

When you have specified your ExperimentState, use the run_experiment function in pipeline.run_experiment that 
will most likely be sufficient for your needs.

Also take a look at pipeline.files for a wide variety of functions that allow you advanced modifications of 
existing experiments without requiring you to fiddle with private variables of sessions yourself.

"""

from copy import deepcopy
from typing import Union, Callable

from alpaca.system.utils.artifact import Artifact
from alpaca.system.interfaces.dataset import Dataset, TrainPoolSet
from alpaca.system.interfaces.model import ModelInterface
from alpaca.system.interfaces.selector import SelectorInterface

from alpaca.system.utils.random_split import random_split
from alpaca.system.utils.random_selector import RandomSelector


class _Factory:
    def __init__(self):
        raise TypeError("Cannot instantiate a class Factory")

    @classmethod
    def keys(cls) -> list[str]:
        return [k for k in cls.__dict__.keys() if not k.startswith("_")]

    @classmethod
    def info(cls) -> None:
        print(f"{cls.__name__} available options ({cls.keys()}): ")
        for k in cls.keys():
            print(f"    {k}")

    @classmethod
    def create(cls, key: str, args: dict) -> object:
        if not key in cls.__dict__:
            raise KeyError(
                f"The requested type {key} cannot be created by factory {cls.__name__}"
            )
        return cls.__dict__[key](**args)

    @classmethod
    def register(cls, key: str, fun: Callable) -> None:
        if not callable(fun):
            raise TypeError("Argument 'fun' must be callable")
        setattr(cls, key, fun)


class DatasetFactory(_Factory):
    pass


class ModelFactory(_Factory):
    pass


class SelectorFactory(_Factory):
    RandomSelector = RandomSelector


class DataState(Artifact):
    def __init__(
        self,
        train_set_ids: list[int] = None,
        pool_set_ids: list[int] = None,
        test_set_ids: list[int] = None,
        dataset_key: str = None,
        dataset_args: dict = dict(),
    ):
        super().__init__()
        self.dataset_key: str = dataset_key
        self.dataset_args: dict = dataset_args
        self.train_set_ids: list[int] = train_set_ids
        self.pool_set_ids: list[int] = pool_set_ids
        self.test_set_ids: list[int] = test_set_ids

    def copy_ids(self, other: "DataState") -> None:
        self.train_set_ids = other.train_set_ids
        self.pool_set_ids = other.pool_set_ids
        self.test_set_ids = other.test_set_ids


class ModelState(Artifact):
    def __init__(
        self,
        model_key: str = None,
        model_args: dict = dict(),
        parameter_file: str = None,
    ):
        super().__init__()
        self.model_key: str = model_key
        self.model_args: dict = model_args
        self.parameter_file: str = parameter_file


class SelectorState(Artifact):
    def __init__(self, selector_key: str = None, selector_args: dict = dict()):
        super().__init__()
        self.selector_key: str = selector_key
        self.selector_args: dict = selector_args


class ExperimentState(Artifact):
    def __init__(
        self,
        data: DataState = None,
        model: ModelState = None,
        selector: SelectorState = None,
        seed: int = 0
    ):
        super().__init__()
        self.data: DataState = deepcopy(data)
        self.model: ModelState = model
        self.selector: SelectorState = selector
        self.seed: int = seed # The seed is used to initialize random number generators at the start of the experiment


class FactoryUtils:
    def __init__(self):
        raise RuntimeError("Create is a namespace class and cannot be instantiated")

    @staticmethod
    def create_data_state(
        data_state: DataState,
    ) -> tuple[TrainPoolSet, Dataset]:
        dataset = DatasetFactory.create(data_state.dataset_key, data_state.dataset_args)
        train_set = dataset.id_select(data_state.train_set_ids)
        pool_set = dataset.id_select(data_state.pool_set_ids)
        test_set = dataset.id_select(data_state.test_set_ids)
        return TrainPoolSet(train_set, pool_set), test_set

    @staticmethod
    def random_split_dataset(
        dataset_key: str,
        dataset_args: dict = dict(),
        train_pool_test_split: Union[tuple[int], tuple[float]] = (0.1, 0.6, 0.3),
        seed: int = 42,
    ) -> DataState:
        dataset = DatasetFactory.create(dataset_key, dataset_args)
        train, pool, test = random_split(dataset, train_pool_test_split, seed=seed)
        return DataState(
            train.ids,
            pool.ids,
            test.ids,
            dataset_key,
            dataset_args,
        )

    @staticmethod
    def create_model_state(model_state: ModelState) -> ModelInterface:
        model = ModelFactory.create(model_state.model_key, model_state.model_args)
        if model_state.parameter_file is not None:
            model.load(model_state.parameter_file)
        return model

    @staticmethod
    def create_selector_state(selector_state: SelectorState) -> SelectorInterface:
        return SelectorFactory.create(
            selector_state.selector_key, selector_state.selector_args
        )

    @staticmethod
    def create_experiment_state(
        experiment_state: ExperimentState,
    ) -> tuple[tuple[TrainPoolSet, Dataset], ModelInterface, SelectorInterface]:
        return (
            FactoryUtils.create_data_state(experiment_state.data),
            FactoryUtils.create_model_state(experiment_state.model),
            FactoryUtils.create_selector_state(experiment_state.selector),
        )

    @staticmethod
    def info() -> None:
        print("\nAvailable experiment components: ")
        DatasetFactory.info()
        ModelFactory.info()
        SelectorFactory.info()
        print()
