from typing import Iterable, Union

import numpy as np

from alpaca.system.interfaces.dataset import DataLoaderInterface


class NumpyDataLoader(DataLoaderInterface):
    def load_batch(self, ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @property
    def x_dims(self) -> int:
        assert len(self) > 0
        x, _ = self.load_batch([0])
        assert len(x.shape) == 2
        return x[0].shape[0]


class NumpyArrayDataLoader(NumpyDataLoader):
    """
    Loads rows from a np.ndarray that has already been loaded into memory.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        self.X = X
        self.y = y

    def load_batch(self, ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(ids, np.ndarray):
            ids = np.array(ids)
        return np.take(self.X, ids, axis=0), np.take(self.y, ids)

    def __len__(self) -> int:
        return self.X.shape[0]
