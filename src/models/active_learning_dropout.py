"""
This file contains a standalone implementation of an Active Learning experiment with Monte Carlo dropout networks. 
"""

import os
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.models.dropout_networks import fit_MC_Dropout_model, fit_concrete_dropout_model

# from data_generator_sp import data_generator
import pandas as pd
import keras

from src.config import DATA_DIR


def get_ranked_uncertainity(
    X_pool: np.ndarray, num_samples: int, model: keras.engine.sequential.Sequential
) -> np.ndarray:
    """calculates for each obervation in X_pool the forecast uncertainty
        of the input model

    Args:
        X_pool (np.ndarray): Observations in the pool
        num_samples (int): number of predictions for each Observation
        model (keras.engine.sequential.Sequential): Keras model for prediction

    Returns:
        np.ndarray(,2): first column contains index for the Observation in X_pool,
                        second column contains the uncertainty of the input model
    """
    uncertainity = np.ones((X_pool.shape[0], 2))

    y_pred_dist_ = [model(X_pool, training=True) for _ in range(num_samples)]
    y_pred_dist = np.concatenate(y_pred_dist_, axis=1)

    uncertainity = np.ones((X_pool.shape[0], 2))
    uncertainity[:, 0] = np.arange(X_pool.shape[0])
    uncertainity[:, 1] = np.std(y_pred_dist, axis=1)
    uncertainity_ = uncertainity[uncertainity[:, 1].argsort()[::-1]]
    return uncertainity_


# Function for doing Active Learning with MC-Network model and query strategy
# with max Uncertainity through MC-Network predictions
def AL_with_MC_NN(
    model_initial,
    test_rmse_initial,
    X_pool,
    y_pool,
    X_train,
    y_train,
    X_test,
    y_test,
    n_queries=20,
    num_samples=25,
    n_points_per_query=10,
):
    # performance_history of Model
    unqueried_score = test_rmse_initial
    performance_history = [unqueried_score]

    for _ in range(n_queries):
        # Rank Prediction-Uncertainitys for the pool-Data
        ranked_uncertainity = get_ranked_uncertainity(
            X_pool, num_samples, model=model_initial
        )
        del model_initial

        # put new (labeled) points according biggest uncertainity in train data
        # pick n_points_per_query
        query_idx = np.array(ranked_uncertainity[:n_points_per_query, 0], dtype=int)
        X_train_new, y_train_new = X_pool[query_idx], y_pool[query_idx]
        X_train = np.concatenate((X_train, X_train_new))
        y_train = np.concatenate((y_train, y_train_new))
        del X_train_new, y_train_new

        # Remove new points from Pool
        X_pool = np.delete(X_pool, query_idx, 0)
        y_pool = np.delete(y_pool, query_idx, 0)
        del query_idx

        # Fit MC-Network with new and old train data set
        model_initial, history = fit_MC_Dropout_model(
            X_train,
            y_train,
            X_test,
            y_test,
            epoch=50,
            dropout_rate=0.045,
            learning_rate=0.001,
        )
        print(_ + 1, ". Query fertig mit query strategie(MC)")
        # Save Models Performance
        test_rmse = history.history["val_root_mean_squared_error"][-1]
        performance_history.append(test_rmse)
        del history, test_rmse
    return performance_history


# Function for doing Active Learning with MC-Network model and random query
# strategy
def AL_with_MC_NN_and_random_sampling(
    test_rmse_initial,
    X_pool,
    y_pool,
    X_train,
    y_train,
    X_test,
    y_test,
    n_queries=20,
    n_points_per_query=10,
):
    # performance_history of Model
    unqueried_score = test_rmse_initial
    performance_history = [unqueried_score]

    for _ in range(n_queries):

        # put new (labeled) points according biggest uncertainity in train data
        # pick n_points_per_query
        query_idx = np.random.choice(
            X_pool.shape[0], size=n_points_per_query, replace=False
        )

        X_train_new, y_train_new = X_pool[query_idx], y_pool[query_idx]
        X_train = np.concatenate((X_train, X_train_new))
        y_train = np.concatenate((y_train, y_train_new))
        del X_train_new, y_train_new

        # Remove new points from Pool
        X_pool = np.delete(X_pool, query_idx, 0)
        y_pool = np.delete(y_pool, query_idx, 0)
        del query_idx

        # Fit MC-Network model with new and old train data set
        model_initial, history = fit_MC_Dropout_model(
            X_train, y_train, X_test, y_test, epoch=50, dropout_rate=0.045
        )
        test_rmse = history.history["val_root_mean_squared_error"][-1]
        print(_ + 1, ". Query fertig mit query strategie(random)")
        # Save Models Performance
        performance_history.append(test_rmse)
        del history, test_rmse
    return performance_history


def main():
    # Create synthetic Data
    # X ,y = data_generator(1000, 1, 2).generate("poly3")
    # plt.figure(dpi=200)
    # plt.scatter(X[:,0], y, s=1)
    # plt.title('Polynomial Degree 3 Function with Noise=2')
    # plt.show()

    # load Featurevectors for ESOL Dataset
    path_esol_feature_vectors = os.path.join(DATA_DIR, "featurevector.csv")
    data_raw = pd.read_csv(path_esol_feature_vectors)

    # normalize Featurevectors
    scaler = StandardScaler().fit(data_raw.drop(["y"], axis=1))
    X = pd.DataFrame(scaler.transform(data_raw.drop(["y"], axis=1)))
    X = np.array(X)
    y = np.array(data_raw.y)

    # split data in train(10%) , pool(60%), test(30%)
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, train_size=int(X.shape[0] * 0.1), random_state=10
    )
    X_test, X_pool, y_test, y_pool = train_test_split(
        X_pool, y_pool, train_size=int(X_pool.shape[0] * (0.3 / 0.9)), random_state=10
    )

    # Fit MC-Network model with inital train data
    model_initial, history = fit_MC_Dropout_model(
        X_train, y_train, X_test, y_test, epoch=50, dropout_rate=0.045
    )
    test_rmse_initial = history.history["val_root_mean_squared_error"][-1]

    # Doing Actrive Learning
    n_queries = 5
    n_points_per_query = 20
    performance_history_MC = AL_with_MC_NN(
        model_initial,
        test_rmse_initial,
        X_pool,
        y_pool,
        X_train,
        y_train,
        X_test,
        y_test,
        n_queries=n_queries,
        num_samples=25,
        n_points_per_query=n_points_per_query,
    )

    performance_history_rand = AL_with_MC_NN_and_random_sampling(
        test_rmse_initial,
        X_pool,
        y_pool,
        X_train,
        y_train,
        X_test,
        y_test,
        n_queries=n_queries,
        n_points_per_query=n_points_per_query,
    )

    # plot RMSE over n_queries
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    ax.plot(performance_history_MC, label="monte carlo uncertainity - sampling")
    ax.plot(performance_history_rand, label="random - sampling")
    ax.scatter(range(len(performance_history_MC)), performance_history_MC, s=13)
    ax.scatter(range(len(performance_history_rand)), performance_history_rand, s=13)
    # ax.set_ylim(bottom=50, top=65)
    ax.grid(True)
    plt.legend(loc="upper left", frameon=True)
    ax.set_title("Incremental RMSE (AL with MC-NeuralNetwork)")
    ax.set_xlabel("Query iteration")
    ax.set_ylabel("RMSE")
    #    plt.show()
    plt.savefig("AL_mit_MC_{}.png".format(round(time.time(), 1)))


if __name__ == "__main__":
    main()
