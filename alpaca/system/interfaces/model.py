from alpaca.system.interfaces.dataset import Dataset
from alpaca.system.utils.artifact import Artifact


class ModelInterface:
    """
    Common interface for models s.t. different model types (pytorch, tensorflow, scikit-learn) can be used interchangeably.

    This should encompass not only the trained weights but also the optimizer. When saving a model to disk to continue training
    again at a later point it is highly recommended to save the optimizer state if the training rate is not constant!
    """

    RequiredDataLoaderType = None

    def __init__(self):
        """
        See derived classes __init__ for arguments.
        """
        raise NotImplementedError()

    @classmethod
    def _check_data_loader_type(cls, dataset: Dataset) -> None:
        if cls.RequiredDataLoaderType is None:
            return
        if not isinstance(dataset.data_loader, cls.RequiredDataLoaderType):
            raise TypeError(
                f"{cls}.fit: Received dataset with data loader of type {type(dataset.data_loader)} "
                f"but requires type {cls.RequiredDataLoaderType}"
            )

    def _derived_fit(self, dataset: Dataset) -> Artifact:
        raise NotImplementedError()

    def fit(self, dataset: Dataset) -> Artifact:
        """
        Fits / trains the model on the given training dataset.

        @param dataset : A dataset compatible with the model type

        @returns : The artifacts recorded during fitting / training
        """
        self._check_data_loader_type(dataset)
        return self._derived_fit(dataset)

    def _derived_RMSE(self, dataset: Dataset) -> float:
        raise NotImplementedError()

    def RMSE(self, dataset: Dataset) -> float:
        """
        Get RMSE for the given dataset.

        @param input : Labeled dataset compatible with the model.

        @returns : Root mean squared error on the input data
        """
        self._check_data_loader_type(dataset)
        return self._derived_RMSE(dataset)

    def save(self, path: str) -> None:
        """
        Save the current model to the given filepath. Depending on the framework this can mean writing the learned parameters, the optimizer state...

        @param path: Filepath
        """
        raise NotImplementedError()

    def load(self, path: str) -> None:
        """
        Load the model state from a file or directly written by the save method...

        @param path: If save produces only one file path is the filename;
        If multiple files need to be written path is the directory to place the files
        """
        raise NotImplementedError()
