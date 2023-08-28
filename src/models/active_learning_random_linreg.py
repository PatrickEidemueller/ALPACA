"""
This file contains a standalone implementation of an active learning baseline with random selection.
As models it uses linear regression, lasso regression and ridge regression.
"""
from os.path import dirname, join

# load packages
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn import linear_model
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
import pandas as pd
import numpy as np
from numpy import arange

# Function for doing Active Learning (learner gets new points, RMSE is tracked)
# ..function returns an Array of RMSE(of the Model) for each query Iteration


def performance_history_doing_AL(X_pool, y_pool, X_test, y_test, learner, n_queries):

    # performance_history of Model
    unqueried_score = mean_squared_error(
        learner.predict(X_test), y_test
    )
    performance_history = [unqueried_score]

    for _ in range(n_queries):

        # get  points from Pool with learner's query strategy
        query_idx, query_instance = learner.query(X_pool)

        # Show learner the new points ( 'ORACLE' )
        X, y = X_pool[query_idx], y_pool[query_idx]
        learner.teach(X=X, y=y)

        # Remove new points from Pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        # Calculate new RMSE
        model_accuracy = mean_squared_error(
            learner.predict(X_test), y_test
        )

        # Save  model's performance
        performance_history.append(model_accuracy)

    return performance_history


# function for random query strategy. it returns the location(index in X_pool)
# and the value of the datapoints, which the learner will get for the next learn-Iteration
# hyperparameter
n_points_per_query = 10


def random_sampling(classifier, X_pool, n_points=n_points_per_query):
    query_idx = np.random.randint(low=0, high=X_pool.shape[0], size=n_points)
    return query_idx, X_pool[query_idx]


# active Learning with lin-Reg and random-query
def main():
    # load data and determine X and y
    featurevectors_path = join(dirname(dirname(dirname(__file__))), "data", "processed", "esol_features_6dims.csv")
    df = pd.read_csv(featurevectors_path)
    X_raw = df.loc[:, df.columns != "y"].to_numpy()
    X_raw = X_raw[:, 1:]
    y_raw = df["y"].to_numpy()

    # train & test split
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=10
    )

    # split data in train(labeled) and pool(unlabeled) data (hyperparameter)
    relative_train_size = 0.25
    X_train, X_pool, y_train, y_pool = train_test_split(
        X_train_pool,
        y_train_pool,
        train_size=int(X_train_pool.shape[0] * relative_train_size),
        random_state=10,
    )

    # Lasso Regression:

    # find alpha
    lasso_alpha = LassoCV(alphas=np.logspace(-3, -1, 30), cv=5).fit(X_raw, y_raw)
    print(type(lasso_alpha))
    print(lasso_alpha)

    # train model
    lasso_model = linear_model.Lasso(
        alpha=lasso_alpha.alpha_, max_iter=100, tol=0.1
    ).fit(X_train, y_train)

    # Ridge Regression:

    # find alpha
    ridgecv = RidgeCV(
        alphas=arange(0.05, 10, 0.05), cv=5, scoring="neg_mean_absolute_error"
    ).fit(X_train, y_train)
    ridgecv.alpha_

    # define model
    ridge_model = linear_model.Ridge(alpha=ridgecv.alpha_, max_iter=100, tol=0.1).fit(
        X_train, y_train
    )

    # fitting learner with train data, linear reg and random_sampling

    learner_random_lr = ActiveLearner(
        estimator=LinearRegression(),
        query_strategy=random_sampling,
        X_training=X_train,
        y_training=y_train,
    )

    learner_random_lasso = ActiveLearner(
        estimator=lasso_model,
        query_strategy=random_sampling,
        X_training=X_train,
        y_training=y_train,
    )

    learner_random_ridge = ActiveLearner(
        estimator=ridge_model,
        query_strategy=random_sampling,
        X_training=X_train,
        y_training=y_train,
    )

    # how many queries ?
    n_queries = int(len(X_pool) / n_points_per_query)

    # doing active learning with pool data and tracking RMSE for each query
    performance_history_lr = performance_history_doing_AL(
        X_pool, y_pool, X_test, y_test, learner_random_lr, n_queries
    )

    performance_history_ridge = performance_history_doing_AL(
        X_pool, y_pool, X_test, y_test, learner_random_ridge, n_queries
    )

    performance_history_lasso = performance_history_doing_AL(
        X_pool, y_pool, X_test, y_test, learner_random_lasso, n_queries
    )

    # plot RMSE over n_queries
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    # plot
    ax.plot(
        performance_history_lr,
        label=f"LR best RMSE: {round(min(performance_history_lr), 2)}",
    )
    ax.plot(
        performance_history_ridge,
        label=f"Ridge best RMSE: {round(min(performance_history_ridge), 2)}",
    )
    ax.plot(
        performance_history_lasso,
        label=f"Lasso best RMSE: {round(min(performance_history_lasso), 2)}",
    )

    ax.scatter(range(len(performance_history_lr)), performance_history_lr, s=4)
    ax.scatter(range(len(performance_history_ridge)), performance_history_ridge, s=4)
    ax.scatter(range(len(performance_history_lasso)), performance_history_lasso, s=4)
    ax.legend()
    ax.grid(True)

    ax.set_title(
        "Linear Regression with random sampling and feature vectors \nIncremental RMSE (AL: {} Trainsize, {} Datapoints per query)".format(
            relative_train_size, n_points_per_query
        )
    )
    ax.set_xlabel("Query iteration")
    ax.set_ylabel("RMSE")
    ax.set_ylim(
        [
            min(
                min(performance_history_lr),
                min(performance_history_ridge),
                min(performance_history_lasso),
            )
            - 0.1,
            max(
                max(performance_history_lr),
                max(performance_history_ridge),
                max(performance_history_lasso),
            )
            + 0.1,
        ]
    )
    plt.show()

    plt.show()

if __name__ == "__main__":
    main()
