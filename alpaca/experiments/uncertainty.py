import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

from alpaca.experiments.config import (
    TRAINING_EPOCH_MAX,
    TRAINING_BATCHSIZE,
    EARLY_STOPPING_UPDATES_STUCK,
    NUM_SAMPLES,
    )
from alpaca.system.interfaces.dataset import TrainPoolSet
from alpaca.system.interfaces.selector import SelectorInterface
from alpaca.system.frameworks.numpy_classes import NumpyDataLoader
from alpaca.system.frameworks.keras_classes import KerasModel
from alpaca.system.factory import (
    ModelFactory,
    SelectorFactory,
    ModelState,
    SelectorState,
)

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(
                        loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def _prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            # n neurons in the VariableLayer since we have n features (kernel_size + bias)
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def _make_uncertainty_arch(
    input_dims: int, num_hidden: int, neurons_per_layer: int, num_training_points: int
) -> tf.keras.Sequential:
    assert num_hidden >= 1
    network = tf.keras.Sequential()
    network.add(tf.keras.Input(shape=(input_dims,)))
    for _ in range(num_hidden):
        network.add(tf.keras.layers.Dense(neurons_per_layer, activation="relu"))
    network.add(
        tfp.layers.DenseVariational(
            1,
            _posterior_mean_field,
            _prior_trainable,
            kl_weight=1 / num_training_points,
        )
    )
    network.add(
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(loc=t, scale=1)
        )
    )
    return network


def _negative_log_likelihood_loss(y_pred: tf.Tensor, y_dist: tf.Tensor) -> tf.Tensor:
    return -y_dist.log_prob(y_pred)


class UncertaintyModel(KerasModel):
    def __init__(self):
        super().__init__(
            max_epoch=TRAINING_EPOCH_MAX,
            training_batchsize=TRAINING_BATCHSIZE,
            early_stopping_updates=EARLY_STOPPING_UPDATES_STUCK)

    def _derived_predict(self, X: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(self.make_samples(X), axis=1)

    def make_samples(self, X: tf.Tensor) -> tf.Tensor:
        """
        Returns 2D-Array of predictions where the i-th row
        contains all the predictions for the i-th datapoint in X.
        """
        samples = [self.network(X) for _ in range(30)]
        return tf.stack(samples, axis=1)

    def save(self, path: str) -> None:
        raise RuntimeWarning(
            f"{type(self).__name__}.save: Skip model saving, as serialization of DenseVariational layers "
            "with custom init arguments may result in an error."
        )


class UncertaintySelector(SelectorInterface):
    RequiredDataLoaderType = NumpyDataLoader
    RequiredModelType = UncertaintyModel

    def __init__(self):
        self.num_samples: int = NUM_SAMPLES

    def _derived_select_batch(
        self, model: UncertaintyModel, train_pool_set: TrainPoolSet, batch_size: int
    ) -> list[int]:
        if len(train_pool_set.pool) <= batch_size:
            return list(range(len(train_pool_set.pool)))
        X, _ = train_pool_set.pool.data_loader.load_batch(train_pool_set.pool.ids)
        samples = model.make_samples(X)
        stddevs: tf.Tensor = tf.squeeze(tf.math.reduce_std(samples, axis=1))
        min_dists: tf.Tensor = None

        # First take point with highest uncertainty
        pool_indices: np.ndarray = np.arange(len(train_pool_set.pool))
        selection = [int(tf.argmax(stddevs))]

        # Now use score which also considers diversity of datapoints
        for _ in range(1, batch_size):
            # Distance to last selected point
            min_dists_last = tf.math.reduce_euclidean_norm(X - X[selection[-1]], axis=1)
            if min_dists is None:
                # First iteration
                min_dists = min_dists_last
            else:
                # New minimum distance is minimum of old minimum and last point distance
                min_dists = tf.reduce_min(
                    tf.stack([min_dists, min_dists_last], axis=1), axis=1
                )

            probabilities, _ = tf.linalg.normalize(
                self._score(stddevs, min_dists), axis=None, ord=1
            )
            distribution = scipy.stats.rv_discrete(
                name="SamplingProb", values=(pool_indices, probabilities.numpy())
            )
            selection.append(distribution.rvs(size=1)[0])
        return selection


class UncertaintySelectorF1Score(UncertaintySelector):
    @staticmethod
    def _score(stddevs: tf.Tensor, dists: tf.Tensor, epistemic_weight=0.5):
        max_val = tfp.stats.percentile(dists, q=90)
        min_val = tfp.stats.percentile(dists, q=20)
        min_val = np.quantile(dists, q=0.2)
        keep_point = tf.logical_and(dists > min_val, dists < max_val)

        # With distance 0 and stddev 0 the likelihood for selection becomes 0
        zeros = tf.zeros_like(stddevs)
        dists = tf.where(keep_point, x=dists, y=zeros)
        stddevs = tf.where(keep_point, x=stddevs, y=zeros)

        dist_scores, _ = tf.linalg.normalize(dists, ord=1)
        epistemic_scores, _ = tf.linalg.normalize(stddevs, ord=1)

        beta = (1.0 - epistemic_weight) / epistemic_weight
        nominator = (beta**2 + 1.0) * epistemic_scores * dist_scores
        denominator = beta**2 * epistemic_scores + dist_scores
        res = nominator / denominator
        return tf.where(tf.math.is_nan(res), x=zeros, y=res)


class UncertaintySelectorQuantileScore(UncertaintySelector):
    @staticmethod
    def _score(stddevs: tf.Tensor, dists: tf.Tensor, percentile: int = 80):
        desired_dist = tfp.stats.percentile(dists, q=percentile)
        epistemic_scores, _ = tf.linalg.normalize(stddevs, ord=1)
        dist_scores = tf.exp(-np.abs(dists - desired_dist))
        return dist_scores * epistemic_scores


class UncertaintyModelSmall(UncertaintyModel):
    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_uncertainty_arch(
            data_loader.x_dims, 2, 64, len(data_loader)
        )
        self.loss = _negative_log_likelihood_loss


class UncertaintyModelMedium(UncertaintyModel):
    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_uncertainty_arch(
            data_loader.x_dims, 3, 256, len(data_loader)
        )
        self.loss = _negative_log_likelihood_loss


class UncertaintyModelWide(UncertaintyModel):
    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_uncertainty_arch(
            data_loader.x_dims, 3, 1024, len(data_loader)
        )
        self.loss = _negative_log_likelihood_loss


class UncertaintyModelDeep(UncertaintyModel):
    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_uncertainty_arch(
            data_loader.x_dims, 10, 256, len(data_loader)
        )
        self.loss = _negative_log_likelihood_loss


ModelFactory.register("UncertaintyModelSmall", UncertaintyModelSmall)
ModelFactory.register("UncertaintyModelMedium", UncertaintyModelMedium)
ModelFactory.register("UncertaintyModelWide", UncertaintyModelWide)
ModelFactory.register("UncertaintyModelDeep", UncertaintyModelDeep)
SelectorFactory.register("UncertaintySelectorF1Score", UncertaintySelectorF1Score)
SelectorFactory.register(
    "UncertaintySelectorQuantileScore", UncertaintySelectorQuantileScore
)


def get_UncertaintyModelSmall() -> ModelState:
    return ModelState(model_key="UncertaintyModelSmall")


def get_UncertaintyModelMedium() -> ModelState:
    return ModelState(model_key="UncertaintyModelMedium")


def get_UncertaintyModelWide() -> ModelState:
    return ModelState(model_key="UncertaintyModelWide")


def get_UncertaintyModelDeep() -> ModelState:
    return ModelState(model_key="UncertaintyModelDeep")


def get_UncertaintySelectorF1() -> SelectorState:
    return SelectorState(
        selector_key="UncertaintySelectorF1Score", selector_args={"num_samples": 20}
    )


def get_UncertaintySelectorQuantile() -> SelectorState:
    return SelectorState(
        selector_key="UncertaintySelectorQuantileScore",
        selector_args={"num_samples": 20},
    )
