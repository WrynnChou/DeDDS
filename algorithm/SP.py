import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from Utils.Utils import eval
from model import get_model

def m_function(x, y):

    if len(x.shape) == 1:
        x = x.reshape((1,x.shape[0]))
    n = x.shape[0]
    dim = x.shape[1]
    N = y.shape[0]
    dim2 = y.shape[1]
    assert dim == dim2, "The dimensions do not match."
    p = np.ones((1, n, 1))
    x_ = p*x.reshape([n,1,dim]) - x
    x_norm = np.linalg.norm(x_, axis=2)
    x_replace, n_zero = replace_zero(x_norm)
    x_inv = np.reciprocal(x_replace)
    part1 = np.zeros((n, dim))
    for i in range(n):
        part1[i,:] = x_inv[i] @ x_[i]
    x2_nrom = np.linalg.norm(x.reshape((n,1,dim)) - y.reshape((1,N,dim)),
                             ord=2, axis=2)
    x2_replace, n_zero = replace_zero(x2_nrom)
    x2_inv = np.reciprocal(x2_replace)
    part2 = np.zeros((n, dim))
    for i in range(n):
        part2[i,:] = x2_inv[i] @ y
    result = np.reciprocal(q(x,y)).reshape((n,1)) * ((N / n) * part1 + part2)
    return result
def q(x, y):

    if len(x.shape) == 1:
        x = x.reshape((1,x.shape[0]))
    n = x.shape[0]
    dim = x.shape[1]
    N = y.shape[0]
    dim2 = y.shape[1]
    assert dim == dim2, "The dimensions do not match."
    x_extend = extend(x, N)
    y_extend = np.array([y] * n)
    diff = x_extend - y_extend
    replace_nrom, n_zero = replace_zero(np.linalg.norm(diff, ord=2, axis=2))
    if n_zero != 0:
        i = 1
    else:
        i = 0
    result = np.sum(np.reciprocal(replace_nrom), axis=1) - i
    return result

def extend(arr, n):
    """
    This function is used to extend arr(n*d) to 3 dimension
    array (n*N*d)
    :param arr: array
    :param n: N
    :return: 3 dimension array (n*N*d)
    """
    if len(arr.shape) == 1:
        arr = arr.reshape((1, arr.shape[0]))
    p = np.ones((1, n, 1))
    b = arr.reshape((arr.shape[0], 1, arr.shape[1]))
    result = b*p

    return result

def replace_zero(arr):
    """
    q function in SP has something wrong, x_i is sampled from y.
    Thus, there must be 0 in x_i - y, whose reciprocal is inf. Here,
    we firstly replace 0 to 1, then sum the reciprocal and minus the
    number of 0.
    :param arr:
    :return:
    """
    arr_cp = arr.copy()
    b = arr == 0
    arr_cp[b] = 1
    n = b.sum()
    return arr_cp, n



class Sp(object):

    def __init__(self, conf, data, label):

        self.conf = conf

        self.n = conf['subsampling_number']

        self.method = conf['method']

        self.data = data

        self.label = label

        self.dim = self.data.shape[1]

        self.num = self.data.shape[0]

        self.model = get_model(self.conf['method'])

    def sampling(self, k = 50, tol=10**(-6)):
        """
        :param k: Iteration
        :param tol: Tolerant
        :return:
        """

        data = self.data
        sample_idx = np.random.choice(self.num, size=self.n, replace=False)
        sample = data[sample_idx, :]
        for i in range(k):
            sample_old = sample
            sample = m_function(sample, data)
            diff = sample - sample_old
            tao = np.linalg.norm(diff, ord=2, axis=1).max()
            if tao <= tol:
                break
        self.tao = tao
        self.sampling = sample
        return tao

    def nnbrs(self):

        data = self.data
        sampling = self.sampling
        y = self.label
        nbr = NearestNeighbors(n_neighbors=1)
        nbr.fit(data)
        dis, indx = nbr.kneighbors(sampling)
        indx = np.ravel(indx)
        subsampling = data[indx,:]
        subsampling_y = y[indx]
        self.subsampling = subsampling
        self.subsampling_y = subsampling_y
        return subsampling, subsampling_y

    def fit(self, train_data=[], train_label=[]):

        model = self.model
        if train_data == []:
            train_data = self.subsampling
            train_label = self.subsampling_y
        model.fit(train_data, train_label)
        self.model = model

    def eval(self, eval_data=[], eval_label=[], fit = True):

        model = self.model
        if eval_data == []:
            eval_data = self.data
            eval_label = self.label
        yp = model.predict(eval_data)
        acc = eval(yp, eval_label, fit)

        return acc
    def plot_2D(self, dim1 = 0, dim2 = 1 ):

        plt.Figure()
        plt.scatter(self.data[:, dim1], self.data[:, dim2])
        plt.scatter(self.subsampling[:, dim1], self.subsampling[:, dim2], c="r")
        plt.show(block=True)

    def plot_2D_sampling(self, dim1 = 0, dim2 = 1 ):

        plt.Figure()
        plt.scatter(self.data[:, dim1], self.data[:, dim2])
        plt.scatter(self.sampling[:, dim1], self.sampling[:, dim2], c="r")
        plt.show(block=True)

