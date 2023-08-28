import numpy as np

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression

from alpaca.system.frameworks.sklearn_classes import SKLLinearModel
from alpaca.system.factory import (
    ModelState,
    ModelFactory,
)


class SKLLinearRegression(SKLLinearModel):
    ModelType = LinearRegression


class SKLLassoRegression(SKLLinearModel):
    ModelType = LassoCV


class SKLRidgeRegression(SKLLinearModel):
    ModelType = RidgeCV


ModelFactory.register("LinearRegression", SKLLinearRegression)
ModelFactory.register("LassoRegression", SKLLassoRegression)
ModelFactory.register("RidgeRegression", SKLRidgeRegression)


def get_LinearRegression() -> ModelState:
    return ModelState(model_key="LinearRegression", model_args={})


def get_LassoRegression() -> ModelState:
    return ModelState(
        model_key="LassoRegression",
        model_args={
            "cv": 5,
            "alphas": np.logspace(-3, -1, 10).tolist(),
            "tol": 0.1,
            "max_iter": 200,
            "random_state": 10,
        },
    )


def get_RidgeRegression() -> ModelState:
    return ModelState(
        model_key="RidgeRegression",
        model_args={"cv": 5, "alphas": np.arange(0.05, 10, 0.5).tolist()},
    )
