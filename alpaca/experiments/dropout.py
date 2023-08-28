import numpy as np
import tensorflow as tf

from concretedropout.tensorflow import (
    ConcreteDenseDropout,
    get_weight_regularizer,
    get_dropout_regularizer,
)

from alpaca.experiments.config import (
    TRAINING_EPOCH_MAX,
    TRAINING_BATCHSIZE,
    EARLY_STOPPING_UPDATES_STUCK,
    NUM_SAMPLES,
    DROPOUT_RATE,
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


def _make_dropout_arch(
    input_dims: int, num_hidden: int, neurons_per_layer: int, dropout_rate: float
) -> tf.keras.Sequential:
    assert num_hidden >= 1
    network = tf.keras.Sequential()
    network.add(tf.keras.Input(shape=(input_dims,)))
    for _ in range(num_hidden):
        network.add(tf.keras.layers.Dropout(dropout_rate))
        network.add(tf.keras.layers.Dense(neurons_per_layer, activation="relu"))
    network.add(tf.keras.layers.Dropout(dropout_rate))
    network.add(tf.keras.layers.Dense(1, activation="linear"))
    network.add(tf.keras.layers.Flatten())
    return network


def _make_concrete_dropout_arch(
    data_loader: NumpyDataLoader,
    num_hidden: int,
    neurons_per_layer: int,
) -> tf.keras.Sequential:
    wr = get_weight_regularizer(len(data_loader), l=1e-2, tau=1.0)
    dr = get_dropout_regularizer(len(data_loader), tau=1.0, cross_entropy_loss=False)

    assert num_hidden >= 1
    network = tf.keras.Sequential()
    network.add(tf.keras.Input(shape=(data_loader.x_dims,)))
    for _ in range(num_hidden):
        network.add(
            ConcreteDenseDropout(
                tf.keras.layers.Dense(neurons_per_layer),
                is_mc_dropout=True,
                weight_regularizer=wr,
                dropout_regularizer=dr,
            )
        )
        network.add(tf.keras.layers.Activation("relu"))
    network.add(
        ConcreteDenseDropout(
            tf.keras.layers.Dense(1),
            is_mc_dropout=True,
            weight_regularizer=wr,
            dropout_regularizer=dr,
        )
    )
    network.add(tf.keras.layers.Activation("linear"))
    network.add(tf.keras.layers.Flatten())
    return network


class DropoutModel(KerasModel):
    """
    Base class for all dropout models.
    """

    def __init__(
        self,
        dropout_rate: float,
        **kwargs
    ):
        self.dropout_rate: float = dropout_rate
        super(DropoutModel, self).__init__(**kwargs)

    def predict_MC(self, X: np.ndarray) -> np.ndarray:
        """
        Passes the input through the model with random dropout applied. This can be used to
        generate multiple predictions for a single datapoint and estimate the model uncertainty
        from their variance.
        """
        return tf.squeeze(self.network(tf.stop_gradient(X), training=True))


class DropoutSelector(SelectorInterface):
    """
    Can be used in combination with a DropoutModel. The selection is solely based on the model's
    uncertainty per datapoint. It does not include a mechanism for batch selection but in practice
    usually still works well for selecting batches.

    The uncertainty of a datapoint is estimated by sampling multiple predictions with random
    dropout applied and then taking the standard deviation. 

    Must be used in combination with a DropoutModel.
    """
    
    def __init__(self, num_samples):
        """
        @param num_samples : refers to the number of samples per datapoint that are used
            for calculating the uncertainty. A higher value gives a more reliable estimation at the
            cost of higher runtime.
        """
        self.num_samples: int = num_samples

    def _derived_select_batch(
        self, model: DropoutModel, train_pool_set: TrainPoolSet, batch_size: int
    ) -> list[int]:
        # Generating num_samples predictions with MC-Network model
        pool_X, _ = train_pool_set.pool.data_loader.load_batch(train_pool_set.pool.ids)
        preds = [model.predict_MC(pool_X) for _ in range(self.num_samples)]
        # Uncertainty is standard deviation of predictions per datapoint
        stacked = tf.stack(preds, axis=1)
        uncertainity = tf.math.reduce_std(stacked, axis=1)

        indices_sorted = tf.argsort(uncertainity)
        return indices_sorted[indices_sorted.shape[0] - batch_size :].numpy().tolist()


class DropoutModelSmall(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_dropout_arch(data_loader.x_dims, 2, 64, self.dropout_rate)


class DropoutModelMedium(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_dropout_arch(data_loader.x_dims, 3, 256, self.dropout_rate)


class DropoutModelWide(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_dropout_arch(
            data_loader.x_dims, 3, 1024, self.dropout_rate
        )


class DropoutModelDeep(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_dropout_arch(
            data_loader.x_dims, 10, 256, self.dropout_rate
        )


class ConcreteDropoutModelSmall(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_concrete_dropout_arch(
            data_loader,
            num_hidden=2,
            neurons_per_layer=64,
        )


class ConcreteDropoutModelMedium(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_concrete_dropout_arch(
            data_loader,
            num_hidden=3,
            neurons_per_layer=256,
        )


class ConcreteDropoutModelWide(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_concrete_dropout_arch(
            data_loader,
            num_hidden=3,
            neurons_per_layer=1024,
        )


class ConcreteDropoutModelDeep(DropoutModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self, data_loader: NumpyDataLoader) -> None:
        self.network = _make_concrete_dropout_arch(
            data_loader,
            num_hidden=10,
            neurons_per_layer=256,
        )


ModelFactory.register("DropoutModelSmall", DropoutModelSmall)
ModelFactory.register("DropoutModelMedium", DropoutModelMedium)
ModelFactory.register("DropoutModelWide", DropoutModelWide)
ModelFactory.register("DropoutModelDeep", DropoutModelDeep)
ModelFactory.register("ConcreteDropoutModelSmall", ConcreteDropoutModelSmall)
ModelFactory.register("ConcreteDropoutModelMedium", ConcreteDropoutModelMedium)
ModelFactory.register("ConcreteDropoutModelWide", ConcreteDropoutModelWide)
ModelFactory.register("ConcreteDropoutModelDeep", ConcreteDropoutModelDeep)
SelectorFactory.register("DropoutSelector", DropoutSelector)


def _make_dropout_model(model_key: str) -> ModelState:
    return ModelState(model_key=model_key, model_args={
        "dropout_rate": DROPOUT_RATE,             
        "max_epoch": TRAINING_EPOCH_MAX,
        "training_batchsize": TRAINING_BATCHSIZE,
        "early_stopping_updates": EARLY_STOPPING_UPDATES_STUCK})


def get_DropoutModelSmall() -> ModelState:
    return _make_dropout_model("DropoutModelSmall")


def get_DropoutModelMedium() -> ModelState:
    return _make_dropout_model("DropoutModelMedium")


def get_DropoutModelWide() -> ModelState:
    return _make_dropout_model("DropoutModelWide")


def get_DropoutModelDeep() -> ModelState:
    return _make_dropout_model("DropoutModelDeep")


def get_ConcreteDropoutModelSmall() -> ModelState:
    return _make_dropout_model("ConcreteDropoutModelSmall")


def get_ConcreteDropoutMedium() -> ModelState:
    return _make_dropout_model("ConcreteDropoutModelMedium")


def get_ConcreteDropoutModelWide() -> ModelState:
    return _make_dropout_model("ConcreteDropoutModelWide")


def get_ConcreteDropoutModelDeep() -> ModelState:
    return _make_dropout_model("ConcreteDropoutModelDeep")


def get_DropoutSelector() -> SelectorState:
    return SelectorState(
        selector_key="DropoutSelector", selector_args={"num_samples": NUM_SAMPLES}
    )
