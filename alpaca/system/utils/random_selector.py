import random

from alpaca.system.interfaces.dataset import TrainPoolSet
from alpaca.system.interfaces.model import ModelInterface
from alpaca.system.interfaces.selector import SelectorInterface


class RandomSelector(SelectorInterface):
    """
    Implements random sampling. Can work with any data loader type and any model type.
    """
    RequiredDataLoaderType = None
    RequiredModelType = None

    def __init__(self, seed: int = 0):
        self._seed = seed

    def _derived_select_batch(
        self, model: ModelInterface, train_pool_set: TrainPoolSet, batch_size: int
    ) -> list[int]:
        indices = list(range(len(train_pool_set.pool)))
        random.seed(self._seed)
        random.shuffle(indices)
        return indices[:batch_size]
