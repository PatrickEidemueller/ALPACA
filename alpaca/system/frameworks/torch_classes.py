from typing import Iterable, Union
from math import sqrt

import torch
from tqdm import tqdm

from alpaca.system.interfaces.dataset import DataLoaderInterface, Dataset
from alpaca.system.interfaces.model import ModelInterface

from alpaca.system.utils.early_stopping import EarlyStoppingStrategy
from alpaca.system.utils.random_split import random_split
from alpaca.system.utils.load_batches import load_batches
from alpaca.system.utils.artifact import ArtifactSeries
from alpaca.system.utils.stopwatch import StopWatch
from alpaca.system.reports import EpochReport


class TorchDataLoader(DataLoaderInterface):
    def load_batch(self, ids: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @property
    def x_dims(self) -> int:
        assert len(self) > 0
        x, _ = self.load_batch([0])
        assert len(x.shape) == 2
        return x[0].shape[0]


class TorchTensorDataLoader(TorchDataLoader):
    """
    Loads rows from a torch.Tensor that is already in memory
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        self.X = X
        self.y = y

    def load_batch(self, ids: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(ids, torch.LongTensor):
            ids = torch.LongTensor(ids)
        return torch.index_select(self.X, 0, ids), torch.index_select(self.y, 0, ids)

    def __len__(self) -> int:
        return self.X.shape[0]


class TorchModel(ModelInterface):
    """
    Can be used in combination with TorchDataLoader.
    """
    RequiredDataLoaderType = TorchDataLoader

    def __init__(
        self,
        max_epoch: int,
        training_batchsize: int,
        use_device: str = None,
        early_stopping_updates: int = None
    ):
        """
        @param max_epoch : Stops training after this number of epochs
        @param training_batchsize : Number of datapoints per minibatch
        @param use_device : Can be either "cpu" or "cuda:<gpu_idx>".
            GPUs are indexed from zero, so if there is only one GPU available on the
            system it is refered to as "cuda:0".
            When it is left None the device will be "cuda:0" if cuda is available
            and "cpu" otherwise.
        @param early_stopping_updates : If early_stopping_updates is None, the model is always trained for max_epochs.
            Otherwise we take 25% of the training set as validation set and stop the training, if the epoch with
            the best validation performance was more than early_stopping_updates parameter updates ago.
            Note that this effectively reduces the training size but can increase training stability by avoiding
            over- and underfitting.
        """
        self._max_epoch: int = max_epoch
        self._stopping_criterion: EarlyStoppingStrategy = None
        if early_stopping_updates is not None:
            self._stopping_criterion: EarlyStoppingStrategy = EarlyStoppingStrategy(
                max_epoch=max_epoch, max_updates_stuck=early_stopping_updates)
            
        self.batchsize: int = training_batchsize

        self._device: torch.device = torch.device("cpu")
        if use_device is not None:
            self._device = torch.device(use_device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")

        self.network: torch.nn.Module = None
        self.optim: torch.optim.Optimizer = None
        self.loss_fun: torch.nn.Module = None

    def _set_model_and_loss(self, training_set: Dataset) -> None:
        """
        Set self.network, self.loss_fun in here. You do not have to move them to a specific device.

        self.network should be the actual neural network deriving from Pytorch's nn.Module
            (see https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html)
        self.loss_fun should be module implementing a loss function deriving from Pytorch's nn.Module
        """
        raise NotImplementedError()

    def _to_device(self, torch_object: object) -> object:
        if isinstance(torch_object, tuple):
            return tuple(self._to_device(x) for x in torch_object)
        return torch_object.to(device=self._device)

    def _derived_fit(self, dataset: Dataset) -> ArtifactSeries:
        """
        Fits / trains the model on the given training dataset.
        @param dataset : A dataset compatible with the model type
        @param epoch : Number of epochs to train
        @param batchsize : Size of minibatches
        returns : The artifacts generated during fitting / training
        """
        self._set_model_and_loss(dataset)
        self.network: torch.nn.Module = self._to_device(self.network)
        self.loss_fun = self._to_device(self.loss_fun)
        self.optim = torch.optim.Adam(params=self.network.parameters(), lr=0.02)
        history = ArtifactSeries()

        if self._stopping_criterion is None:
            for i in tqdm(range(self._max_epoch)):
                report = self._fit_one_epoch(dataset)
                report.epoch = i
                history.append(report)
            return history
        
        # Use early stopping strategy
        self._stopping_criterion.reset()
        train_set, validation_set = random_split(dataset, [0.75, 0.25])
        while not self._stopping_criterion.stopping_criterion_met:
            report = self._fit_one_epoch(train_set)
            
            validation_rmse = self._derived_RMSE(validation_set)
            self._stopping_criterion.update(
                {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                },
                validation_loss=validation_rmse,
                num_updates=len(train_set)/self.batchsize)

            report.epoch = self._stopping_criterion.current_epoch
            report.validation_size = len(validation_set)
            report.validation_rmse = validation_rmse
            history.append(report)
        
        print(f"    Training stopped after {self._stopping_criterion.current_epoch} / {self._stopping_criterion.max_epoch} epochs")
        checkpoint = self._stopping_criterion.best_model_weights
        self.network.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optim.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        return history
        
    def _fit_one_epoch(self, trainset: Dataset) -> EpochReport:
        with StopWatch() as stopwatch:
            epoch_loss = 0
            for X, y in load_batches(
                trainset,
                batchsize=self.batchsize,
                shuffle=True,
                transform=self._to_device,
            ):
                self.optim.zero_grad()
                y_pred = self.network(X)
                y_pred = torch.squeeze(y_pred)
                loss = self.loss_fun(y_pred, torch.squeeze(y))
                loss.backward()
                self.optim.step()
                epoch_loss += float(loss)
            return EpochReport(
                    train_size=len(trainset),
                    train_loss=epoch_loss,
                    minibatch_size=self.batchsize,
                    timestamp_start=stopwatch.start,
                    duration=stopwatch.elapsed(),
                )

            

    def _derived_RMSE(self, dataset: Dataset) -> float:
        # If feature vectors have at most 2000 dims == 8kB and preload=8 we use at max
        # 4096 * 8000 * 8 = 260 MB of memory, adjust if necessary
        ssd = 0.0
        with torch.no_grad():
            for X, y in load_batches(
                dataset=dataset,
                batchsize=2**13,
                shuffle=False,
                preload=8,
                transform=self._to_device,
            ):
                y_pred = self.network(X)
                y_pred = torch.squeeze(y_pred)
                # float64 should always be large enough as long as the squared errors are not larger than approx e+290
                ssd += float(torch.sum((y_pred - y) ** 2))
        return sqrt(ssd / len(dataset))

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        self.network.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optim.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
