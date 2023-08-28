"""
The code was mostly taken from https://github.com/JordanAsh/badge and slightly modified. 

MIT License

Copyright (c) 2021 Natural Language Processing @UCLA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from alpaca.system.files import DATA_DIR
from os.path import join

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb
import gc
from copy import copy, deepcopy
from scipy import stats
import numpy as np

# from sklearn.externals.six import string_types
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import KBinsDiscretizer


class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = torch.from_numpy(X).to(torch.float32)
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


def get_esol(path, binnig_strategy="kmeans", nr_optimal_bins=100):
    data = pd.read_csv(path)
    X = data.drop("y", axis=1)
    y = data["y"]

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = X_tr.copy()
    print("Loaded esol, discretization")
    if binnig_strategy == "quantile":
        classes = KBinsDiscretizer(
            n_bins=nr_optimal_bins, encode="ordinal", strategy="quantile"
        ).fit_transform(Y_tr.values.reshape(-1, 1))
    elif binnig_strategy == "kmeans":
        classes = (
            KMeans(
                n_clusters=nr_optimal_bins,
                random_state=42,
                init="k-means++",
                n_init="auto",
            )
            .fit(X_tr)
            .labels_
        )
    else:
        classes = KBinsDiscretizer(
            n_bins=nr_optimal_bins, encode="ordinal", strategy="uniform"
        ).fit_transform(Y_tr.values.reshape(-1, 1))

    train_data["cluster_id"] = classes
    train_data["y"] = Y_tr
    cluster_to_mean = train_data.groupby("cluster_id")["y"].mean().to_dict()

    X_tr = X_tr.values
    X_te = X_te.values

    Y_tr = np.array(classes)
    Y_te = Y_te.values

    Y_tr = Y_tr.reshape((Y_tr.shape[0],))
    Y_tr = torch.from_numpy(Y_tr)

    Y_te = Y_te.reshape((Y_te.shape[0],))
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te, cluster_to_mean


class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    # training for one single epoch
    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        losses = []

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):

            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()

            # forward function of the model is called
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            losses.append(loss.detach().numpy())
            # compute sum of correctly classified examples of the batch
            loss.backward()

            for p in filter(lambda p: p.grad is not None, self.clf.parameters()):
                p.grad.data.clamp_(min=-0.1, max=0.1)
            optimizer.step()

        # return the accuracy over the entire dataset (1 epoch), and the current loss
        return np.mean(losses)

    # training process for several epochs

    def train(self, reset=True, optimizer=0, verbose=True, data=[], net=[]):
        def weight_reset(m):
            newLayer = deepcopy(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.manual_seed(10)
                m.reset_parameters()

        n_epoch = self.args["n_epoch"]
        if reset:
            self.clf = self.net.apply(weight_reset)
        if type(net) != list:
            self.clf = net
        if type(optimizer) == int:
            optimizer = optim.Adam(
                self.clf.parameters(), lr=self.args["lr"], weight_decay=0
            )

        # either get the data using the parameter data of type list from this function or get the data from the constructor
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(
            self.handler(
                self.X[idxs_train],
                torch.Tensor(self.Y.numpy()[idxs_train]).long(),
            ),
            shuffle=True,
            **self.args["loader_tr_args"],  # Batchsize 32
        )
        if len(data) > 0:
            loader_tr = DataLoader(
                self.handler(
                    data[0],
                    torch.Tensor(data[1]).long(),
                ),
                shuffle=True,
                **self.args["loader_tr_args"],
            )

        epoch = 1
        while True:
            lossCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            if epoch >= 101:
                break

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(
                self.handler(X, Y),
                shuffle=False,
                **self.args["loader_te_args"],  # Batchsize = 64
            )
        else:
            loader_te = DataLoader(
                self.handler(X.numpy(), Y),
                shuffle=False,
                **self.args["loader_te_args"],  # Batchsize = 64
            )

        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x), Variable(y)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    # gradient embedding for badge (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model=[]):
        if type(model) == list:
            model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(
            self.handler(X, Y),
            shuffle=False,
            **self.args["loader_te_args"],
        )
        with torch.no_grad():
            for x, y, idxs in loader_te:

                x, y = Variable(x), Variable(y)
                cout, out = model(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            # idxs[j] is the index for a particular observation in the pool set
                            embedding[idxs[j]][
                                embDim * c : embDim * (c + 1)
                            ] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][
                                embDim * c : embDim * (c + 1)
                            ] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)


# X is a array of gradient for each observation in pool set and K is the batch size
def init_centers(X: np.ndarray, K: int):
    # compute the norm (norm of order 2) of each gradient in the poolset and get the index/indices of the highest gradients
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    # get the highest gradients and store them in a python list
    mu = [X[ind]]
    # indsAll includes the indices of selected samples from the poolset, first element will be the index/indices with the highest gradients
    indsAll = [ind]
    # list of zeros with the pool set size
    centInds = [0.0] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')

    while len(mu) < K:

        # compute euclidian distance of each gradient and highest gradient
        # D2 is a vector with the distances of each gradient with the highest gradient if len(mu) == 1
        # D2 is a vector with the lowest distances of each gradient in poolset with respect to the gradients in the batch
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:

            # we always consider the lowest distance of a particular gradient with respect to the current batch
            # newD includes the distances of gradients in poolset to the latest added gradient to the batch
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)

        # square the distances and divide by the sum of squared to get probabilities
        Ddist = (D2**2) / sum(D2**2)

        # examples with higher distance will have higher probability (diverse batch)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))

        # generate one example/index using the specified distribution

        np.random.seed(10)
        ind = customDist.rvs(size=1)[0]

        # while ind in indsAll: ind = customDist.rvs(size=1)[0]
        # add the selected gradient from the poolset to the mu list
        mu.append(X[ind])
        # add the selected index from the poolset to the indsAll list
        indsAll.append(ind)
        cent += 1
    return indsAll


class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super().__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]
        ).numpy()
        chosen = init_centers(gradEmbedding, n)
        return idxs_unlabeled[chosen]


class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super().__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        inds = np.where(self.idxs_lb == 0)[0]
        np.random.seed(10)
        return inds[np.random.permutation(len(inds))][:n]


def active_learning(
    path=join(DATA_DIR, "processed", "esol_pca_100dims.csv"),
    model="mlp",
    binning_strategy="quantile",
    nClasses=100,
    sampling_strategy="rand",
    NUM_QUERY=50,
    NUM_INIT_LB=50,
):

    # data defaults
    args_pool = {
        "ESOL": {
            "n_epoch": 10,
            "transform": None,
            "loader_tr_args": {"batch_size": 32, "num_workers": 0},
            "loader_te_args": {"batch_size": 64, "num_workers": 0},
            "lr": 1e-4,
            "optimizer_args": {
                "lr": 0.01,
                "momentum": 0.5,
            },
        }
    }

    args_pool["ESOL"]["transformTest"] = args_pool["ESOL"]["transform"]
    args = args_pool["ESOL"]

    # path to feature vectors

    X_tr, Y_tr, X_te, Y_te, cluster_to_mean = get_esol(path, binning_strategy, nClasses)

    dim = X_tr.shape[1]
    nEmb = 128

    class linMod(nn.Module):
        def __init__(self, dim=28):
            super().__init__()
            self.dim = dim
            self.lm = nn.Linear(dim, nClasses)

        def forward(self, x):
            out = self.lm(x)
            return out, x

        def get_embedding_dim(self):
            return self.dim

    class mlpMod(nn.Module):
        def __init__(self, dim, embSize=128):
            super(mlpMod, self).__init__()
            self.embSize = embSize
            self.dim = dim
            self.lm1 = nn.Linear(self.dim, embSize)
            self.lm2 = nn.Linear(embSize, embSize)

            self.linear = nn.Linear(embSize, nClasses, bias=False)

        def forward(self, x):
            output1 = F.relu(self.lm1(x))
            output2 = F.relu(self.lm2(output1))
            out = self.linear(output2)
            return out, output2  # output2 (grad emb vorletzte Schicht)

        def get_embedding_dim(self):
            return self.embSize

    if model == "mlp":
        net = mlpMod(dim, embSize=nEmb)
    elif model == "lin":
        net = linMod(dim=dim)
    else:
        print("choose a valid model - mlp, resnet, or vgg")
        raise ValueError

    handler = DataHandler

    n_pool = len(Y_tr)
    n_test = len(Y_te)
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.seed(5)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

    if type(X_tr[0]) is not np.ndarray:
        X_tr = X_tr.numpy()

    if sampling_strategy == "rand":
        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif sampling_strategy == "badge":
        strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)

    NUM_ROUND = int((X_tr.shape[0] - NUM_INIT_LB) / NUM_QUERY)

    strategy.train()
    P = strategy.predict(X_te, Y_te).numpy()
    preds = [cluster_to_mean[pred] for pred in P]
    first_mse = np.sqrt(np.sum((preds - Y_te.numpy()) ** 2) / len(preds))
    scores = [first_mse]

    for rd in range(1, NUM_ROUND + 1):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # query
        output = strategy.query(NUM_QUERY)
        q_idxs = output
        idxs_lb[q_idxs] = True

        # update
        strategy.update(idxs_lb)
        strategy.train(verbose=False)

        P = strategy.predict(X_te, Y_te).numpy()
        preds = [cluster_to_mean[pred] for pred in P]
        rmse = np.sqrt(np.sum((preds - Y_te.numpy()) ** 2) / len(preds))

        scores.append(rmse)

    return scores, NUM_ROUND


def plot_result(batch_size_list=[50, 100, 150]):

    for batch_size in batch_size_list:
        for model in ["lin", "mlp"]:
            for binning_strategy in ["quantile", "kmeans"]:

                random_rmses, n_rounds = active_learning(
                    sampling_strategy="rand",
                    model=model,
                    NUM_QUERY=batch_size,
                    binning_strategy=binning_strategy,
                )
                badge_rmses, n_rounds = active_learning(
                    sampling_strategy="badge",
                    model=model,
                    NUM_QUERY=batch_size,
                    binning_strategy=binning_strategy,
                )

                queries = range(0, n_rounds + 1)
                fig, ax = plt.subplots()

                ax.plot(queries, random_rmses, label="Random")
                ax.plot(queries, badge_rmses, label="Badge")

                ax.set_xlabel("Nr of Queries")
                ax.set_ylabel("RMSE")
                plt.title(
                    f"Random (Blue) vs Badge (Orange), BinningStrategy {binning_strategy}, Model {model}, BatchSize {batch_size}"
                )
                plt.savefig(
                    f"Random vs Badge, BinningStrategy {binning_strategy}, Model {model}, BatchSize {batch_size}"
                )
                plt.legend()


if __name__ == "__main__":
    plot_result()
