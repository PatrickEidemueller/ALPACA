"""
This file contains code for comparing different discretization strategies of a continuous target variable.
The purpose of this was that originally we wanted to used the BADGE implementation from https://github.com/JordanAsh/badge.
This implementation is only for classification problems, so in order to use it for our use case at Sanofi we
discretized the target variable.

Later we discovered, that there is an algorithm for regression problems which is equivalent to BADGE included in the
bmdal framework.
"""

from os.path import dirname, join

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans


def tune_bins(
    path="featurevectors.csv",
    binning_strategy="kmeans",
    nr_bins=list(range(20, 100, 5)),
):
    data = pd.read_csv(path)
    X = data.drop("y", axis=1)
    y = data["y"].values

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = X_tr.copy()
    train_data["y"] = Y_tr
    bin_nn_mse = {}

    def model(classes):
        train_data["class_id"] = classes
        bins_to_mean = train_data.groupby("class_id")["y"].mean().to_dict()

        nn = MLPClassifier(
            activation="relu",
            alpha=0.00001,
            hidden_layer_sizes=(100, 100),
            random_state=42,
            max_iter=900,
        ).fit(X_tr, train_data["class_id"])

        pred_nn = [bins_to_mean[pred] for pred in nn.predict(X_te)]

        return np.sqrt(mean_squared_error(Y_te, pred_nn))

    if binning_strategy == "quantile":
        for i in nr_bins:
            discretized_y = KBinsDiscretizer(
                n_bins=i, encode="ordinal", strategy="quantile"
            ).fit_transform(Y_tr.reshape(-1, 1))
            bin_nn_mse[i] = model(discretized_y)

    elif binning_strategy == "kmeans":
        for i in nr_bins:
            kmeans_labels = (
                KMeans(n_clusters=100, random_state=42, init="k-means++", n_init="auto")
                .fit(X_tr)
                .labels_
            )
            bin_nn_mse[i] = model(kmeans_labels)

    elif binning_strategy == "uniform":
        for i in nr_bins:
            discretized_y = KBinsDiscretizer(
                n_bins=i, encode="ordinal", strategy="uniform"
            ).fit_transform(Y_tr.reshape(-1, 1))
            bin_nn_mse[i] = model(discretized_y)

    return min(bin_nn_mse, key=bin_nn_mse.get)


def main():
    featurevectors_path = join(dirname(dirname(dirname(__file__))), "data", "processed", "esol_mol2vec_100dims.csv")
    result = tune_bins(
        featurevectors_path, "quantile", [2, 4, 6, 8, 10]
    )
    print(result)


if __name__ == "__main__":
    main()
