import numpy as np
import matplotlib.pyplot as plt
from Utils.Utils import eval
from model import get_model


def partition(arr, l, r, idx):
    x = arr[r]
    i = l
    for j in range(l, r):

        if arr[j] <= x:
            arr[i], arr[j] = arr[j], arr[i]
            idx[i], idx[j] = idx[j], idx[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    idx[i], idx[r] = idx[r], idx[i]
    return i, idx

def kthSmallest(arr, l, r, k, idx):

    if (k > 0 and k <= r - l + 1):

        index, idx = partition(arr, l, r, idx)

        if (index - l == k - 1):
            return arr[0 : (index + 1)], idx[0 : (index + 1)]

        if (index - l > k - 1):
            return kthSmallest(arr, l, index - 1, k, idx)

        return kthSmallest(arr, index + 1, r,
                           k - index + l - 1,
                           idx)
    print("Index out of bound")

def kthBiggest(arr, l, r, k, idx):
    arr_inv = -arr
    value, index = kthSmallest(arr_inv, l, r, k, idx)
    return -value, index

class Iboss(object):

    def __init__(self, conf, data, label):

        self.conf = conf

        self.n = conf['subsampling_number']

        self.method = conf['method']

        self.data = data

        self.label = label

        self.dim = self.data.shape[1]

        self.num = self.data.shape[0]

        self.model = get_model(self.conf['method'])

    def sampling(self):

        idx = set({})
        dim = self.dim
        data = self.data
        label = self.label
        p = np.ceil(self.n/(2*dim))
        assert self.dim <= self.n, "the dimension of data is too high, which is bigger than subsampling size."
        idx_init = np.arange(self.num)
        for i in range(dim):

            arr1 = data[:, i].copy()
            arr2 = data[:, i].copy()
            idx_temp1 = idx_init.copy()
            idx_temp2 = idx_init.copy()
            kbig, idx_b = kthBiggest(arr1, 0, (len(arr1) -1), p, idx_temp1)
            ksmall, idx_s = kthSmallest(arr2, 0, (len(arr2) -1), p, idx_temp2)
            idx = idx | set(idx_s) | set(idx_b)
            data = np.delete(self.data, np.array(list(idx)), axis=0)
            idx_init = np.delete(np.arange(self.num), np.array(list(idx)))

            if len(idx) >= self.n:
                break

        idx_a = np.array(list(idx))
        idx_a = idx_a[0:self.n]
        sampling_data = self.data[idx_a, :]
        sampling_label = label[idx_a]
        self.subsampling = sampling_data
        self.subsampling_y = sampling_label
        self.subsampling_idx = idx_a
        return sampling_data, sampling_label

    def fit(self, train_data=[], train_label=[]):

        model = self.model
        if train_data == []:
            train_data = self.subsampling
            train_label = self.subsampling_y
        model.fit(train_data, train_label)
        self.model = model

    def eval(self, eval_data = [], eval_label = []):

        model = self.model
        if eval_data == []:

            eval_data = self.data
            eval_label = self.label

        yp = model.predict(eval_data)
        acc = eval(yp, eval_label)

        return acc

    def plot_2D(self, dim1 = 0, dim2 = 1 ):

        plt.Figure()
        plt.scatter(self.data[:, dim1], self.data[:, dim2])
        plt.scatter(self.subsampling[:, dim1], self.subsampling[:, dim2], c="r")
        plt.show(block=True)