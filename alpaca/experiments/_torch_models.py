import torch

from alpaca.system.factory import ModelFactory, ModelState
from alpaca.system.interfaces.dataset import Dataset
from alpaca.system.frameworks.torch_classes import TorchModel, TorchDataLoader

from alpaca.experiments.config import TRAINING_EPOCH_MAX, TRAINING_BATCHSIZE, EARLY_STOPPING_UPDATES_STUCK


def make_arch(
    n_inputs: int, n_outputs: int, n_hidden_layers: int, neurons_per_layer: int
) -> torch.nn.Module:
    if n_hidden_layers == 0:
        return torch.nn.Linear(n_inputs, n_outputs)

    network = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, neurons_per_layer),
        torch.nn.ReLU(),
    )
    for _ in range(n_hidden_layers - 1):
        network.append(
            torch.nn.Linear(neurons_per_layer, neurons_per_layer),
        )
        network.append(
            torch.nn.ReLU(),
        )
    network.append(
        torch.nn.Linear(neurons_per_layer, n_outputs),
    )
    return network


class TorchModelSmall(TorchModel):
    """
    Two hidden layers with 256 neurons each
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_model_and_loss(self, training_set: Dataset) -> None:
        data_loader: TorchDataLoader = training_set.data_loader
        self.network = make_arch(
            n_inputs=data_loader.x_dims,
            n_outputs=1,
            n_hidden_layers=2,
            neurons_per_layer=64,
        )
        self.loss_fun = torch.nn.MSELoss()


class TorchModelMedium(TorchModel):
    """
    Two hidden layers with 256 neurons each
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_model_and_loss(self, training_set: Dataset) -> None:
        data_loader: TorchDataLoader = training_set.data_loader
        self.network = make_arch(
            n_inputs=data_loader.x_dims,
            n_outputs=1,
            n_hidden_layers=3,
            neurons_per_layer=256,
        )
        self.loss_fun = torch.nn.MSELoss()


class TorchModelWide(TorchModel):
    """
    Two hidden layers with 1024 neurons each
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_model_and_loss(self, training_set: Dataset) -> None:
        data_loader: TorchDataLoader = training_set.data_loader
        self.network = make_arch(
            n_inputs=data_loader.x_dims,
            n_outputs=1,
            n_hidden_layers=3,
            neurons_per_layer=1024,
        )
        self.loss_fun = torch.nn.MSELoss()


class TorchModelDeep(TorchModel):
    """
    Ten hidden layers with 256 neurons each
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_model_and_loss(self, training_set: Dataset) -> None:
        data_loader: TorchDataLoader = training_set.data_loader
        self.network = make_arch(
            n_inputs=data_loader.x_dims,
            n_outputs=1,
            n_hidden_layers=10,
            neurons_per_layer=256,
        )
        self.loss_fun = torch.nn.MSELoss()


ModelFactory.register("TorchModelSmall", TorchModelSmall)
ModelFactory.register("TorchModelMedium", TorchModelMedium)
ModelFactory.register("TorchModelWide", TorchModelWide)
ModelFactory.register("TorchModelDeep", TorchModelDeep)


def _get_model_state(key: str) -> ModelState:
    return ModelState(
        model_key=key, model_args=dict(
        max_epoch=TRAINING_EPOCH_MAX,
        training_batchsize=TRAINING_BATCHSIZE,
        early_stopping_updates=EARLY_STOPPING_UPDATES_STUCK))


def get_TorchModelSmall() -> ModelState:
    return _get_model_state("TorchModelSmall")


def get_TorchModelMedium() -> ModelState:
    return _get_model_state("TorchModelMedium")


def get_TorchModelWide() -> ModelState:
    return _get_model_state("TorchModelWide")


def get_TorchModelDeep() -> ModelState:
    return _get_model_state("TorchModelDeep")
