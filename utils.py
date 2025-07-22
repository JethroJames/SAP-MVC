import scipy.io as scio
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as Fun
import torch.nn as nn
from types import SimpleNamespace

def load_data(name, views):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))

    X = []
    for i in range(0, views):
        tmp = data['X' + str(i + 1)]
        tmp = tmp.astype(np.float32)
        X.append(torch.from_numpy(tmp).to(dtype=torch.float))

    return X, labels

def random_split(X, Y, train_size=0.7):
    Y = torch.tensor(Y)
    number_class = torch.unique(Y)
    index_train = []
    index_test = []
    for i in range(0, number_class.size(0)):
        indices = torch.nonzero(torch.eq(Y, number_class[i])).squeeze()
        random_indices = torch.randperm(len(indices)).tolist()
        indices_train = random_indices[0:int(train_size * len(indices))]
        indices_test = random_indices[int(train_size * len(indices)):]
        index_train.extend(indices[indices_train])
        index_test.extend(indices[indices_test])
    X_train = []
    X_test = []
    for i in range(0, len(X)):
        X_train.append(X[i][index_train, :])
        X_test.append(X[i][index_test, :])
    Y_train = Y[index_train]
    Y_test = Y[index_test]
    return X_train, X_test, Y_train, Y_test


def distance(X, Y, square=True):
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))
    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)
