from copy import copy

import torch

from alpaca.system.interfaces.dataset import TrainPoolSet, Dataset, DataLoaderInterface
from alpaca.system.frameworks.torch_classes import TorchDataLoader, TorchTensorDataLoader
from alpaca.system.frameworks.numpy_classes import NumpyDataLoader, NumpyArrayDataLoader


def _get_converted_data_loader(
    data_loader: DataLoaderInterface, requested_type: type
) -> "DataLoaderInterface":
    """
    Converts the dataset to the requested type if possible. Otherwise throws NotImplementeError.
    If container is already of the correct type, returns the container.

    @param container : The container to be converted.
    @param requested_type : Requested type of the converted container. Must be a subclass of
        ContainerInterface.

    @returns : Container of the requested type
    """
    if isinstance(DataLoaderInterface, requested_type):
        return data_loader
    # Do conversion
    if requested_type == NumpyDataLoader:
        if isinstance(data_loader, TorchTensorDataLoader):
            return NumpyArrayDataLoader(data_loader.X.numpy(), data_loader.y.numpy())
    if requested_type == TorchDataLoader:
        if isinstance(data_loader, NumpyArrayDataLoader):
            return TorchTensorDataLoader(
                torch.from_numpy(data_loader.X), torch.from_numpy(data_loader.y)
            )
    raise NotImplementedError(
        f"convert_container: Conversion from {type(data_loader)} to {requested_type} has not been implemented, yet."
    )


def convert_dataset(dataset: Dataset, requested_data_loader_type: type) -> None:
    """
    Converts the dataset to the requested type if possible. Otherwise throws error.
    If dataset is already of the correct type, returns the dataset.

    @param dataset : The dataset to be converted.
    @param requested_data_loader_type : Requested ContainerType of the converted dataset.
        Must be a subclass of ContainerInterface.

    @returns : Dataset with the requested container type
    """
    if requested_data_loader_type is None:
        return
    if isinstance(dataset.data_loader, requested_data_loader_type):
        return
    dataset.data_loader = _get_converted_data_loader(
        dataset.data_loader, requested_data_loader_type
    )


def convert_train_pool_set(
    tp_set: TrainPoolSet, requested_data_loader_type: type
) -> None:
    """
    Converts the train_pool_set to the requested type if possible. Otherwise throws error.
    If dataset is already of the correct type, does nothing.

    @param dataset : The dataset to be converted.
    @param requested_data_loader_type : Requested ContainerType of the converted dataset.
        Must be a subclass of ContainerInterface.

    @returns : Dataset with the requested container type
    """
    if requested_data_loader_type is None:
        return
    if isinstance(tp_set.data_loader, requested_data_loader_type):
        return tp_set
    data_loader = _get_converted_data_loader(
        tp_set.train.data_loader, requested_data_loader_type
    )
    tp_set._train_set.data_loader = data_loader
    tp_set._pool_set.data_loader = data_loader
