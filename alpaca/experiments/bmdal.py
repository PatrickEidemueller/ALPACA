import torch
from bmdal_reg.bmdal import algorithms, feature_data

from alpaca.system.factory import (
    ModelState,
    SelectorState,
    SelectorFactory,
)
from alpaca.system.interfaces.dataset import TrainPoolSet
from alpaca.system.interfaces.selector import SelectorInterface
from alpaca.system.frameworks.torch_classes import TorchModel, TorchDataLoader

SIGMA = 0.1
OVERSELECTION_FACTOR = 1.0


class BMDALSelector(SelectorInterface):
    """
    The BMDAL selector allows using 'bmdal_reg' in the pipeline.
    It simply invokes 'bmdal_reg.bmdal.algorithms.select_batch' with the arguments passed to its constructor.

    For more info on 'bmdal_reg' see:
    https://github.com/dholzmueller/bmdal_reg

    The source code of 'bmdal_reg.bmdal.algorithms.select_batch' is accessible under this link:
    https://github.com/dholzmueller/bmdal_reg/blob/main/bmdal_reg/bmdal/algorithms.py

    The BMDALSelector can only be used in combination with a TorchModel.
    """
    RequiredModelType = TorchModel
    RequiredDataLoaderType = TorchDataLoader

    def __init__(self, **select_batch_args):
        """
        @param select_batch_args : The arguments to pass to bmdal_reg.bmdal.algorithms.select_batch
            The full list of arguments can be found in the docstring of the function:
            https://github.com/dholzmueller/bmdal_reg/blob/main/bmdal_reg/bmdal/algorithms.py
            Note that "data" and "model" are already filled in by the BMDALSelector automatically.
        """
        if "data" in select_batch_args:
            raise RuntimeError(
                "BMDALSelector: The 'data' argument of select_batch must not be specified!"
            )
        if "model" in select_batch_args:
            raise RuntimeError(
                "BMDALSelector: The 'model' argument of select_batch must not be specified!"
            )
        self.kwargs = select_batch_args

    def _derived_select_batch(
        self, model: TorchModel, train_pool_set: TrainPoolSet, batch_size: int
    ) -> list[int]:
        # This loads all of train and pool into memory. If that is a problem
        # this method has to be rewritten to work on a subset of the pool set
        X_train, y_train = train_pool_set.data_loader.load_batch(
            train_pool_set.train.ids
        )
        y_train = torch.reshape(y_train, (-1, 1))
        X_pool, _ = train_pool_set.data_loader.load_batch(train_pool_set.pool.ids)
        new_indices, _ = algorithms.select_batch(
            models=[model.network],
            y_train=y_train,
            data={
                "train": feature_data.TensorFeatureData(model._to_device(X_train)),
                "pool": feature_data.TensorFeatureData(model._to_device(X_pool)),
            },
            batch_size=batch_size,
            **self.kwargs,
        )
        return new_indices.tolist()


SelectorFactory.register("BMDALSelector", BMDALSelector)


def _get_bmdal_experiment(selector_args: dict) -> SelectorState:
    return SelectorState(selector_key="BMDALSelector", selector_args=selector_args)


def get_bmdal_random() -> SelectorState:
    selector_args = dict(
        selection_method="random", base_kernel="linear", kernel_transforms=[]
    )
    return _get_bmdal_experiment(selector_args=selector_args)


def get_bmdal_maxdiag_grad_rp_512_acs_rf_512() -> SelectorState:
    selector_args = dict(
        selection_method="maxdiag",
        base_kernel="grad",
        kernel_transforms=[("rp", [512]), ("acs-rf", [512, SIGMA, None])],
    )
    return _get_bmdal_experiment(selector_args=selector_args)


def get_bmdal_maxdet_p_grad_rp_512_train() -> SelectorState:
    selector_args = dict(
        selection_method="maxdet",
        base_kernel="grad",
        kernel_transforms=[("rp", [512]), ("train", [SIGMA, None])],
    )
    return _get_bmdal_experiment(selector_args=selector_args)


def get_bmdal_bait_f_p_grad_rp_512_train() -> SelectorState:
    selector_args = dict(
        selection_method="bait",
        overselection_factor=OVERSELECTION_FACTOR,
        base_kernel="grad",
        kernel_transforms=[("rp", [512]), ("train", [SIGMA, None])],
    )
    return _get_bmdal_experiment(selector_args=selector_args)


def get_bmdal_fw_p_ll_acs_rf_hyper_512() -> SelectorState:
    selector_args = dict(
        selection_method="fw",
        base_kernel="ll",
        kernel_transforms=[("acs-rf-hyper", [512, None])],
    )
    return _get_bmdal_experiment(selector_args=selector_args)


def get_bmdal_maxdist_p_grad_rp_512_train() -> SelectorState:
    selector_args = dict(
        selection_method="maxdist",
        sel_with_train=False,
        base_kernel="grad",
        kernel_transforms=[("rp", [512]), ("train", [SIGMA, None])],
    )
    return _get_bmdal_experiment(selector_args=selector_args)


def get_bmdal_kmeanspp_p_grad_rp_512_acs_rf_512() -> SelectorState:
    selector_args = dict(
        selection_method="kmeanspp",
        sel_with_train=False,
        base_kernel="grad",
        kernel_transforms=[("rp", [512]), ("acs-rf", [512, SIGMA, None])],
    )
    return _get_bmdal_experiment(selector_args=selector_args)


def get_bmdal_lcmd_tp_grad_rp_512() -> SelectorState:
    selector_args = dict(
        selection_method="lcmd", base_kernel="grad", kernel_transforms=[("rp", [512])]
    )
    return _get_bmdal_experiment(selector_args=selector_args)
