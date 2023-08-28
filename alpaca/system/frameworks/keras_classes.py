from math import sqrt
from typing import Union
from copy import deepcopy

import tensorflow as tf
from tqdm import tqdm

from alpaca.system.interfaces.dataset import Dataset
from alpaca.system.interfaces.model import ModelInterface

from alpaca.system.utils.early_stopping import EarlyStoppingStrategy
from alpaca.system.utils.random_split import random_split
from alpaca.system.utils.load_batches import load_batches
from alpaca.system.utils.stopwatch import StopWatch
from alpaca.system.utils.artifact import ArtifactSeries
from alpaca.system.reports import EpochReport
from alpaca.system.frameworks.numpy_classes import NumpyDataLoader


class KerasModel(ModelInterface):
    """
    Can be used in combination with NumpyDataLoader, the conversion to torch tensors happens implicitly.
    """
    RequiredDataLoaderType = NumpyDataLoader

    def __init__(self,         
        max_epoch: int = None,
        training_batchsize: int = None,
        early_stopping_updates: int = None):
        """
        @param max_epoch : Stops training after this number of epochs
        @param training_batchsize : Number of datapoints per minibatch
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

        self.network: tf.keras.Model = None
        self.optimizer: tf.optimizers.Optimizer = tf.optimizers.Adam(learning_rate=0.02)
        self.loss: Union[tf.losses.Loss, str] = tf.losses.MeanSquaredError()

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        """
        Set network, optim and loss in here. The network should implement the keras Models API
        (see https://keras.io/api/models/)

        Do not compile it, as this is done when _derived_fit is called.
        """
        raise NotImplementedError()

    def _derived_RMSE(self, dataset: Dataset) -> float:
        # If feature vectors have at most 2000 dims == 8kB and preload=8 we use at max
        # 4096 * 8000 * 8 = 260 MB of memory, adjust if necessary
        ssd = 0.0
        for X, y in load_batches(dataset, batchsize=2**13, preload=8):
            y_pred = tf.squeeze(self._derived_predict(tf.stop_gradient(X)))
            # float64 should always be large enough as long as the squared errors are not larger than approx e+290
            ssd += float(tf.keras.backend.sum((y_pred - tf.stop_gradient(y)) ** 2))
        return sqrt(ssd / len(dataset))

    def _derived_predict(self, X: tf.Tensor) -> tf.Tensor:
        """
        Needs to be overridden by some classes like Bayesian Networks
        which have non-deterministic output and require taking the mean
        of multiple samples.
        """
        return self.network(X, training=False)

    def _derived_fit(self, dataset: Dataset) -> ArtifactSeries:
        """
        Fits / trains the model on the given training dataset.

        @param dataset : A dataset compatible with the model type
        @param epoch : Number of epochs to train
        @param batchsize : Size of minibatches

        @returns : The artifacts generated during fitting / training
        """
        self._init_model(dataset.data_loader)
        self.network.compile(optimizer=self.optimizer, loss=self.loss)
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
                deepcopy(self.network.get_weights()),
                validation_loss=validation_rmse,
                num_updates=len(train_set)/self.batchsize)

            report.epoch = self._stopping_criterion.current_epoch
            report.validation_size = len(validation_set)
            report.validation_rmse = validation_rmse
            history.append(report)
        
        print(f"    Training stopped after {self._stopping_criterion.current_epoch} / {self._stopping_criterion.max_epoch} epochs")
        self.network.set_weights(
            deepcopy(self._stopping_criterion.best_model_weights),
        )
        return history

    def _fit_one_epoch(self, trainset: Dataset) -> EpochReport:
        with StopWatch() as stopwatch:
            epoch_loss = 0
            for X, y in load_batches(trainset, batchsize=self.batchsize):
                loss = self.network.train_on_batch(X, y)
                epoch_loss += loss
            return EpochReport(
                    train_size=len(trainset),
                    train_loss=epoch_loss,
                    minibatch_size=self.batchsize,
                    timestamp_start=stopwatch.start,
                    duration=stopwatch.elapsed(),
                )

    def save(self, path: str) -> None:
        if self.network is None:
            raise RuntimeWarning(
                f"{type(self).__name__}.save: Model has not been created yet!"
            )
        self.network.save(path, save_format="h5")

    def load(self, path: str) -> None:
        self.network = tf.keras.models.load_model(path)
