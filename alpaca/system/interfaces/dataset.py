"""
This contains classes for loading and interacting with data.

The three relevant classes are:
    - DataLoaderInterface: Abstract base class for any classes that actually
        load and provide data.
    - Dataset: Wraps DataLoader and allows a framework independent way to create subsets
     of existing datasets. For more info on why this additional layer is needed see section "Ids" below. 
    - TrainPoolSet: Simulates labeling of observations

Ids:
    In order to allow lightweight serialization of observations we require that every
    observation has an index and an id. The index depends on the Dataset that
    the observation is contained in while the id depends on the DataLoader (which can
    be shared by many Datasets). This is best explained with an example:

    Let's say our DataLoader can access a database of 5 observations:
        0: x=(0.230, 3.948) y=1.048
        1: x=(8.235, 6.293) y=3.238
        2: x=(2.3549, -1.88) y=0.55
        3: x=(2.015, 0.293) y=0.89
        4: x=(4.45, -3.673) y=2.94
    Then the ids of this database are [0, 1, 2, 3, 4]
    
    At the start we create a Dataset from the DataLoader so the mapping from
    index to observation is simply
        idx=0  -> 0: x=(0.230, 3.948) y=1.048
        idx=1 -> 1: x=(8.235, 6.293) y=3.238
        idx=2 -> 2: x=(2.3549, -1.88) y=0.55
        idx=3 -> 3: x=(2.015, 0.293) y=0.89
        idx=4 -> 4: x=(4.45, -3.673) y=2.94

    We now randomly split the dataset indices into two subsets:
        [idx=2, idx=4] [idx=3, idx=1, idx=0]
    Now we create the subsets with Dataset.index_select():
    The first subset looks like:
        idx=0 -> 2: x=(2.3549, -1.88) y=0.55
        idx=1-> 4: x=(4.45, -3.673) y=2.94
    The second subset looks like:
        idx=0 -> 3: x=(2.015, 0.293) y=0.89
        idx=1 -> 1: x=(8.235, 6.293) y=3.238
        idx=2  -> 0: x=(0.230, 3.948) y=1.048

    While the index is affected by selecting subsets of a DataSet the id is unchanged.
    This means that both subsets can still use the same underlying DataLoader and we can
    create subsets of our datasets without having to implement them for the specific type
    of DataLoader (i.e. framework).

    Also by saving the id instead of the index, we can easily recreate a certain Dataset
    as long as we use the same DataLoader. This requires that a DataLoader is deterministic,
    i.e. always loads the observations in the same order!
"""

from typing import Iterable


class DataLoaderInterface:
    """
    Abstract interface class for data loaders of labeled datasets.

    To implement for specific framework.
        1. Derive your own class (e.g. named NumpyDataLoader if your data is loaded as np.ndarray)
        2. In your derived class override the 'load_batch' and '__len__' methods

    """

    def __init__(self):
        """
        For arguments see derived class.
        """
        raise NotImplementedError()

    def load_batch(self, ids: Iterable[int]) -> tuple[object, object]:
        """
        Load the observations at the specified ids from the database. The DataLoader must always return the same
        observation for the same id.

        This must also hold between between different runs of the program.
        So if your data loader has some kind of random behaviour, make sure to require
        a seed as argument in the __init__ function.

        @param id : Id of the observation to load. Will be in range(0, len(self))

        @returns : An tuple of objects (X, y) that contain the unlabeled training data and the ground truth labels.
            The types of X any y depend on the framework.
            Typically X is a 2d tensor / array where the i-th row is the "X" part corresponding to the i-th id and y
            is a 1d tensor / array where the i-th value is the "y" part.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Number of rows in the container.

        @returns : Number of rows in the container.
        """
        raise NotImplementedError()


def _check_data_loader_equal(dl1, dl2) -> None:
    if dl1 != dl2:
        raise TypeError(
            "Expected two references to the same dataloader but data loaders "
            f"are of types {type(dl1)}, {type(dl2)} and are objects at the "
            f"following addresses (python id()) {id(dl1)}, {id(dl2)}"
        )


class Dataset:
    """
    A labeled dataset. It that maps indices to ids of the underlying data loader.

    Do not derive from it, this is the exact behaviour of a Dataset
    that we support in the pipeline.
    """

    def __init__(self, data_loader: DataLoaderInterface):
        """
        @param data_loader : data loader that the dataset should use.
        """
        if not (isinstance(data_loader, DataLoaderInterface)):
            raise TypeError(
                f"Dataset.__init__: Argument 'data_loader' must derive "
                f"from DataLoaderInterface but was of type {type(data_loader)}"
            )
        self.data_loader = data_loader
        self._ids = list(range(len(self.data_loader)))

    def __add__(self, other: "Dataset") -> "Dataset":
        """
        Combines the observations of self and other.

        @param other : Must use the same data loader (exact same instance, not just the same type)

        @returns : Concatenated dataset (observations from right appended to left).
        """
        _check_data_loader_equal(self.data_loader, other.data_loader)
        new_dataset = Dataset(self.data_loader)
        new_dataset._ids = self._ids + other._ids
        return new_dataset

    def __len__(self) -> int:
        """
        Number of observations in the dataset.

        @returns : Number of observations in the dataset.
        """
        return len(self._ids)

    def index_select(self, indices: list[int]) -> "Dataset":
        """
        Returns a dataset that only contains the observations at the given indices of the current dataset.

        @param indices : Indices that should be selected.

        @returns : Dataset of the same type as the dataset it is called on.
        """
        if type(indices) != list:
            raise TypeError(
                f"Dataset.index_select: indices must be list, but were of type {type(indices)}"
            )
        new_dataset = Dataset(self.data_loader)
        new_dataset._ids = [self._ids[i] for i in indices]
        return new_dataset

    def index_exclude(self, indices: list[int]) -> "Dataset":
        """
        Returns a dataset that contains all except the observations at the given indices of the current dataset.

        @param indices : Indices that should be excluded.

        @returns : Dataset using the same data loader as the dataset it is called on.
        """
        if type(indices) != list:
            raise TypeError(
                f"Dataset.index_exclude: indices must be list, but were of type {type(indices)}"
            )
        complement_indices = list(set(range(len(self))) - set(indices))
        return self.index_select(complement_indices)

    def id_select(self, ids: list[int]) -> "Dataset":
        """
        Similar to index_select but selects the observations by id instead

        @param ids : IDs of the observations that shall be selected.

        @returns : Dataset using the same data loader as the dataset it is called on.
        """
        dataset = Dataset(self.data_loader)
        dataset._ids = ids
        return dataset

    @property
    def ids(self) -> list[int]:
        """
        @returns : List where the i-th entry is a unique and persistent id for the i-th
            observation in the dataset.
        """
        return self._ids


class TrainPoolSet:
    """
    Encapsulates the train and pool set.
    """

    def __init__(self, train_set: Dataset, pool_set: Dataset):
        _check_data_loader_equal(train_set.data_loader, pool_set.data_loader)
        self._train_set = train_set
        self._pool_set = pool_set

    @property
    def pool(self) -> Dataset:
        return self._pool_set

    @property
    def train(self) -> Dataset:
        return self._train_set

    @property
    def data_loader(self):
        return self._train_set.data_loader

    def label_datapoints(self, indices: Iterable) -> Dataset:
        """
        Simulates the 'labeling' step in active learning. Adds the observations at the given
        indices of the pool set to the train set. Then removes them from the pool set.

        @param indices : Indices to be labeled. Must be valid indices of the current pool set

        @returns : Dataset with only the newly labeled observations
        """
        labeled_observations = self._pool_set.index_select(indices)
        self._train_set += labeled_observations
        self._pool_set = self._pool_set.index_exclude(indices)
        return labeled_observations
