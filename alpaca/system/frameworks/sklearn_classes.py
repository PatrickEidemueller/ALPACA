from math import sqrt

import pickle
import numpy as np
from sklearn.linear_model._base import LinearModel

from alpaca.system.interfaces.dataset import Dataset
from alpaca.system.interfaces.model import ModelInterface
from alpaca.system.frameworks.numpy_classes import NumpyArrayDataLoader
from alpaca.system.utils.artifact import Artifact
from alpaca.system.utils.stopwatch import StopWatch
from alpaca.system.utils.load_batches import load_batches


class SKLLinearModel(ModelInterface):
    """
    Base class for scikit learn linear regression models.

    Requires data to fit into memory completely i.e. can only be used with NumpyArrayDataLoader.
    """

    RequiredContainerType = NumpyArrayDataLoader
    ModelType: type[LinearModel] = None

    def __init__(self, **kwargs):
        self.args = kwargs
        self.model: LinearModel = self.ModelType(**kwargs)

    def _derived_fit(self, dataset: Dataset, **training_args: dict) -> "Artifact":
        # Loads all the training data into main memory!
        X, y = dataset.data_loader.load_batch(dataset.ids)
        with StopWatch() as stopwatch:
            self.model = self.model.fit(X, y)
            duration = stopwatch.elapsed()
        return Artifact(duration=duration)

    def _derived_RMSE(self, dataset: Dataset) -> float:
        ssd = 0.0
        for X, y in load_batches(dataset, batchsize=2**14, preload=4):
            ssd = float(np.sum((y - self.model.predict(X)) ** 2))
        return sqrt(ssd / len(dataset))

    def save(self, path: str) -> None:
        """
        Save the current model to the given filepath. Depending on the framework this can mean writing the learned parameters, the optimizer state...

        @param path : Filepath
        """
        with open(path, "wb") as file:
            pickle.dump((self.model, self.args), file)

    def load(self, path: str) -> None:
        """
        Load the model state from a file or directly written by the save method...

        @param path : If save produces only one file path is the filename.
            If multiple files need to be written path is the directory to place the files
        """
        with open(path, "rb") as file:
            model, args = pickle.load(file)
        self.model = model
        self.args = args
