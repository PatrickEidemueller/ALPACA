import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from numpy.linalg import matrix_power
from scipy.spatial.distance import cdist

# parameters: nsamples, nfeatures, noise
# generate linear regression data with noise, logistic data with noise, polynom data with noise
class data_generator:
    def __init__(self, nsamples: int, nfeatures: int, noise: float, seed: int):
        self.nsamples = nsamples
        self.nfeatures = nfeatures
        self.noise = noise
        self.seed = seed

    def generate_linear(self) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        return make_regression(
            n_samples=self.nsamples, n_features=self.nfeatures, noise=self.noise
        )
    
    def generate_logistic(self):
        x, y = self.generate_linear()
        return x, 1 / (1 + np.exp(-y))

    def generate_centers(self, num_centers: int):
        x, _ = self.generate_linear()
        center_idxs = np.random.choice(np.arange(x.shape[0]), size=num_centers)
        centers = x[center_idxs, :]
        #vals = np.random.sample(size=num_centers)
        vals = np.linspace(-1, 1, num_centers)
        distmat = cdist(x, centers)
        distmat = distmat / np.reshape(np.linalg.norm(distmat, axis=1), (distmat.shape[0], 1))
        valmat = distmat * vals
        y = np.sum(valmat, axis=1)
        y = y + self.noise * np.random.randn(y.shape[0])
        return x, y

if __name__=="__main__":
    x, y = data_generator(1000, 2, 0.01, 0).generate_centers(20)
    plt.scatter(x=x[:,0], y=x[:,1], c=y, s=1)
    plt.show()
    y = np.reshape(y, (y.shape[0], 1))
    data = np.concatenate([x, y], axis=1)
    df = pd.DataFrame(data, columns=["1", "2", "y"])
    df.to_csv("synthetic_dataset2D.csv", ",", index=None)