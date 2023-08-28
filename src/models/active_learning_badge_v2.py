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

# load packages
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset
import gc
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from copy import copy as copy
from copy import deepcopy as deepcopy
from scipy import stats
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# function to map the mean of y in the bins to the df to later calculate RMSE
def get_bins_mean(path):
    data = pd.read_csv(path)
    X = data.drop("y", axis=1)
    y = data["y"].values
    y = y.reshape((y.shape[0], 1))

    # train test split to find bins
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    df = X_tr.copy()
    discretized_y = KBinsDiscretizer(
        n_bins=nClasses, encode="ordinal", strategy="quantile", random_state=40
    ).fit_transform(Y_tr.reshape(-1, 1))
    df["bin_id"] = discretized_y
    df["y"] = Y_tr
    bins_to_mean = df.groupby("bin_id")["y"].mean().to_dict()

    return bins_to_mean


# function to get the dataset and transform the data in train size  and test size variables,
# it transforms X in torch tensor size ([1125, 100])
# further more transform Y_tr and Y_te in torch tensors and X_tr and X_te in ndarrays train size (900, 100) and test size (225, 100)


def get_esol(path):
    data = pd.read_csv(path)
    X = data.drop("y", axis=1).values
    y = data["y"].values

    y = y.reshape((y.shape[0], 1))

    # discretizing y in 10 quantile bins
    # discretizer = KBinsDiscretizer(n_bins= nClasses, encode='ordinal', strategy= binning_strategy)
    # discretized_y = discretizer.fit_transform(y)

    # train test split to find bins
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # creating the number of bins according to the best fit for our nn
    Y_tr = KBinsDiscretizer(
        n_bins=nClasses, encode="ordinal", strategy="quantile", random_state=40
    ).fit_transform(Y_tr.reshape(-1, 1))

    # Y input has to be a tensor
    Y_tr = Y_tr.reshape((Y_tr.shape[0],))
    Y_tr = torch.from_numpy(Y_tr)

    # Y input has to be a tensor
    Y_te = Y_te.reshape((Y_te.shape[0],))
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te


# discretizing y in 10 quantile bins
# discretizer = KBinsDiscretizer(n_bins= nClasses, encode='ordinal', strategy= binning_strategy)
# discretized_y = discretizer.fit_transform(y)

#  discretizing y in optimal quantile bins


def get_bins(path, binning_strategy):

    data = pd.read_csv(path)
    X = data.drop("y", axis=1)
    y = data["y"].values
    y = y.reshape((y.shape[0], 1))

    # train test split to find bins
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    nr_bins = list(range(5, 100, 5))
    train_data = X_tr.copy()
    bin_nn_mse = {"nn": {}}
    if binning_strategy == "quantile":

        nr_bins = list(range(5, 100, 5))
        train_data = X_tr.copy()
        bin_nn_mse = {"nn": {}}
        for i in nr_bins:
            discretized_y = KBinsDiscretizer(
                n_bins=i, encode="ordinal", strategy="quantile", random_state=40
            ).fit_transform(Y_tr.reshape(-1, 1))
            train_data["bin_id"] = discretized_y
            train_data["y"] = Y_tr

            bins_to_mean = train_data.groupby("bin_id")["y"].mean().to_dict()

            nn = MLPClassifier(
                activation="relu",
                alpha=0.00001,
                hidden_layer_sizes=(100, 100),
                random_state=42,
                max_iter=900,
            ).fit(X_tr, train_data["bin_id"])

            pred_nn = [bins_to_mean[pred] for pred in nn.predict(X_te)]

            bin_nn_mse["nn"][i] = mean_squared_error(Y_te, pred_nn)

        nClasses = min(bin_nn_mse["nn"], key=bin_nn_mse["nn"].get)

    elif binning_strategy == "kmeans":
        nr_bins = list(range(150, 300, 10))
        for i in nr_bins:
            kmeans_labels = (
                KMeans(n_clusters=i, random_state=42, init="k-means++")
                .fit(X_tr)
                .labels_
            )
            train_data["bin_id"] = kmeans_labels
            train_data["y"] = Y_tr

            bins_to_mean = train_data.groupby("bin_id")["y"].mean().to_dict()

            nn = MLPClassifier(
                activation="relu",
                alpha=0.00001,
                hidden_layer_sizes=(100, 100),
                random_state=42,
                max_iter=900,
            ).fit(X_tr, train_data["bin_id"])

            pred_nn = [bins_to_mean[pred] for pred in nn.predict(X_te)]

            bin_nn_mse["nn"][i] = mean_squared_error(Y_te, pred_nn)

        nClasses = min(bin_nn_mse["nn"], key=bin_nn_mse["nn"].get)

    else:
        print("Not a valid strategy")

    return nClasses


# Data Handler class which is subclass of Dataset and is used to store the training data
# and fetch a batch from our trainignset by calling the function get_item
class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        # X input has to be a tensor
        self.X = torch.from_numpy(X).to(torch.float32)
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


# The Strategy class consists of the following attributes: X, Y, idxs_lb (with True labels used to select the data for the active training loop),
# the length of Y n_pool, the net which is used for training, the handler which is used to store the training data
# and args a dictionary which contains information about training, testing and data transformation
class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        # use_cuda = torch.cuda.is_available()

    # the function query takes as input the batch_size and returns the indices of selected/ queried data from the pool set
    def query(self, n):
        pass

    # the function update updates the idxs_lb attribute by setting True values in positions of the selected observations from the pool set
    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    # training for one single epoch
    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()  # clf = variable name for classifier
        losses = []

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x), Variable(y)
            optimizer.zero_grad()
            # forward function of the model is called
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            losses.append(loss.detach().numpy())
            loss.backward()
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()):
                p.grad.data.clamp_(min=-0.1, max=0.1)
            optimizer.step()
        # return loss
        return np.mean(losses)  # loss.item()

    # training process for several epochs
    def train(self, reset=True, optimizer=0, verbose=True, data=[], net=[]):
        def weight_reset(m):
            newLayer = deepcopy(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
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
                transform=self.args["transform"],
            ),
            shuffle=True,
            **self.args["loader_tr_args"],
        )
        if len(data) > 0:
            loader_tr = DataLoader(
                self.handler(
                    data[0],
                    torch.Tensor(data[1]).long(),
                    transform=self.args["transform"],
                ),
                shuffle=True,
                **self.args["loader_tr_args"],
            )
        epoch = 1
        while True:
            lossCurrent = self._train(epoch, loader_tr, optimizer)
            # print(f'Epoch : {epoch} --- training Loss : {lossCurrent}')
            epoch += 1
            if epoch > 50:
                break

    # remove cuda() and reshape prediction
    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(
                self.handler(X, Y, transform=self.args["transformTest"]),
                shuffle=False,
                **self.args["loader_te_args"],
            )
        else:
            loader_te = DataLoader(
                self.handler(X.numpy(), Y, transform=self.args["transformTest"]),
                shuffle=False,
                **self.args["loader_te_args"],
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
            self.handler(X, Y, transform=self.args["transformTest"]),
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


# from query_strategies.py
# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax(
        [np.linalg.norm(s, 2) for s in X]
    )  # erster zentroi = größter gradient andere je größer distanz zu ausgewählten gradient
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0

    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:

            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)

        Ddist = (D2**2) / sum(D2**2)

        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))

        np.random.seed(10)
        ind = customDist.rvs(size=1)[0]

        # while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]
        ).numpy()
        chosen = init_centers(gradEmbedding, n)
        return idxs_unlabeled[chosen]


class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        inds = np.where(self.idxs_lb == 0)[0]
        np.random.seed(10)
        return inds[np.random.permutation(len(inds))][:n]


class mlpMod(nn.Module):
    def __init__(self, dim, embSize=128, useNonLin=True):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = dim  # dim = X_tr.shape[1]
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, embSize)
        self.linear = nn.Linear(embSize, nClasses, bias=False)

    def forward(self, x):
        out1 = F.relu(self.lm1(x))
        out2 = F.relu(self.lm2(out1))
        out = self.linear(out2)
        # out are the output logits (10 logits since we have 10 classes) and emb are the activations
        # of last layer (size of embSize)
        return out, out2  # output2 (grad emb vorletzte Schicht)

    def get_embedding_dim(self):
        return self.embSize


# from run.py start run:
NUM_INIT_LB = 50  # X variables
NUM_QUERY = 50  #
DATA_NAME = "ESOL"  # feature vectors
torch.manual_seed(0)
np.random.seed(5)

# data defaults
args_pool = {
    "ESOL": {
        "n_epoch": 10,
        "transform": None,
        "loader_tr_args": {"batch_size": 32, "num_workers": 0},
        "loader_te_args": {"batch_size": 64, "num_workers": 0},
        "optimizer_args": {
            "lr": 0.01,
            "momentum": 0.5,
        },
    },
}
args_pool["ESOL"]["transformTest"] = args_pool["ESOL"]["transform"]
args = args_pool[DATA_NAME]

# path to feature vectors
path = "/Users/patrickeidemuller/Documents/Uni/MasterDataScience/3_ProjektSanofi/featurevector.csv"

# define classes for regression dependet on bins
binning_strategy = "kmeans"  # kmeans or quantile

# get optimal bins and define the output classes
# nClasses = get_bins(path, binning_strategy)
nClasses = 280  # (for strategy quantile) 280 for kmeans
# print(optimal_bins)

# get the mean of Y for the appropiate bin
bins_mean = get_bins_mean(path)

# load dataset
X_tr, Y_tr, X_te, Y_te = get_esol(path)

dim = X_tr.shape[1]

handler = DataHandler

LR = 0.0001
MODEL = "mlp"
LAMB = 1

args["lr"] = LR
args["modelType"] = MODEL
args["lamb"] = LAMB

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print("number of labeled pool: {}".format(NUM_INIT_LB))
print("number of unlabeled pool: {}".format(n_pool - NUM_INIT_LB))
print("number of testing pool: {}".format(n_test))

# idxs_lb is
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# load specified network
nEmb = 128
if MODEL == "mlp":
    net = mlpMod(dim, embSize=nEmb)
else:
    print("choose a valid model")
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

# set up the specified sampler
strategy_random = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
strategy_badge = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
if type(X_te) == torch.Tensor:
    X_te = X_te.numpy()

# round 0 accuracy
NUM_ROUND = int((X_tr.shape[0] - NUM_INIT_LB) / NUM_QUERY)

# dictionary for saving bins mean and prediciton
bin_nn_rmse = {"Random": {}, "badge": {}}

# train the model with badge sampler
strategy_badge.train()
P = strategy_badge.predict(X_te, Y_te)

pred_nn = [bins_mean[pred] for pred in P.detach().numpy()]
first_loss = np.sqrt(((pred_nn - Y_te.detach().numpy()) ** 2).mean())
print(
    f"Number of Training Points to Start : {NUM_INIT_LB} -- testing RMSE Badge : {first_loss}"
)

bin_nn_rmse["badge"][0] = first_loss

for rd in range(1, NUM_ROUND + 1):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # query
    output = strategy_badge.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True

    # update
    strategy_badge.update(idxs_lb)
    strategy_badge.train(verbose=False)

    # round accuracy
    P = strategy_badge.predict(X_te, Y_te)
    pred_nn = [bins_mean[pred] for pred in P.detach().numpy()]
    # bin_nn_rmse['nn'][rd] = np.sqrt(mean_squared_error(Y_te, pred_nn))
    bin_nn_rmse["badge"][rd] = np.sqrt(((pred_nn - Y_te.detach().numpy()) ** 2).mean())
    print(f'Round : {rd} -- Testing RMSE BADGE: {bin_nn_rmse["badge"][rd]}')

# train the model with random sampler
strategy_random.train()
P = strategy_random.predict(X_te, Y_te)

pred_nn = [bins_mean[pred] for pred in P.detach().numpy()]
first_loss = np.sqrt(((pred_nn - Y_te.detach().numpy()) ** 2).mean())
print(
    f"Number of Training Points to Start : {NUM_INIT_LB} -- Testing RMSE Random: {first_loss}"
)

bin_nn_rmse["Random"][0] = first_loss

for rd in range(1, NUM_ROUND + 1):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # query
    output = strategy_random.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True

    # update
    strategy_random.update(idxs_lb)
    strategy_random.train(verbose=False)

    # round accuracy
    P = strategy_random.predict(X_te, Y_te)

    pred_nn = [bins_mean[pred] for pred in P.detach().numpy()]
    # bin_nn_rmse['nn'][rd] = np.sqrt(mean_squared_error(Y_te, pred_nn))
    bin_nn_rmse["Random"][rd] = np.sqrt(((pred_nn - Y_te.detach().numpy()) ** 2).mean())
    print(f'Round : {rd} -- Testing RMSE RANDOM: {bin_nn_rmse["Random"][rd]}')

# plot the loss
queries = range(0, NUM_ROUND + 1)
fig, ax = plt.subplots()

ax.plot(queries, bin_nn_rmse["Random"].values(), label="Random_rmse")
ax.plot(queries, bin_nn_rmse["badge"].values(), label="Badge_rmse")

ax.set_xlabel("Number of Queries")
ax.set_ylabel("RMSE")
ax.grid(True)
plt.title(
    f"Random vs. Badge (Nr. of Classes: {nClasses}, Strategy: {binning_strategy}, Labeled Pool: {NUM_INIT_LB}, Data Points per Query: {NUM_QUERY})"
)
plt.legend()
plt.show()
