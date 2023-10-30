import matplotlib.pyplot as plt
import numpy as np
import torch
from disropt.agents import Agent
import Utils.Utils as uts
from model import get_model

class DeOptLinReg(object):
    """
    Distributed Subdata Selection for Big Data via Sampling-Based Approach
    """
    def __init__(self, conf, data, label, comm):

        self.agent = None

        self.conf = conf

        self.n = conf['subsampling_number']

        self.method = conf['method']

        self.data = data

        self.label = label

        self.dim = self.data.shape[1]

        self.num = self.data.shape[0]

        self.model = get_model(self.conf['method'])

        self.comm = comm

        self._rank = self.rank

        self._size = self.size

    @property
    def rank(self):
        return self.comm.Get_rank()

    @property
    def size(self):
        return self.comm.Get_size()

    def pre_sampling(self, r: int = 0):
        """
        Step 1 uniformly random sampling
        :param r: presampling number
        :return: subdata
        """
        if r == 0:
            r = self.n
        # note that self.n is the sampling size on one client, self.n * self.size is the total size.
        r_0 = int(r)
        index = np.random.randint(0, self.num, r_0)
        sampling = self.data[index, :]
        sampling_y = self.label[index]

        self.pre_samp = sampling
        self.pre_samp_y = sampling_y
        return sampling, sampling_y

    def pilot_estimator(self):
        """
        Pilot estimator from pre sampling
        :return: pilot estimator \hat{\beta_0}
        """

        combine_x = np.array(self.comm.gather(self.pre_samp, root=0))
        combine_y = np.array(self.comm.gather(self.pre_samp_y, root=0))
        if self.rank == 0:
            x_ = np.concatenate(combine_x)
            y_ = np.concatenate(combine_y)
            beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_), x_)), np.transpose(x_)), y_)
        else:
            beta = 0
        beta = self.comm.bcast(beta, root=0)
        self.beta_0 = beta.reshape((7,1))
        return self.beta_0

    def calculate_pi(self, c: float = 10**(-6)):
        """
        step 1 calculate probability pi and sampling size rk
        :param c:
        :return: p_{ik}, r_k
        """

        ui = np.zeros(self.num)
        for i in range(self.num):
            ui[i] = np.maximum(np.abs(self.label[i] - np.matmul(np.transpose(self.beta_0), self.data[i, :])), c) *\
                 np.linalg.norm(self.data[i, :])
        U0 = ui.sum()
        pi_i = ui / U0
        U = self.comm.reduce(U0)
        U = self.comm.bcast(U)
        rk_ = int(np.around((self.n * U0 * self.size) / U))
        self.pi = pi_i
        self.rk = rk_
        return pi_i, rk_

    def sampling(self):
        """
        Step 2 sampling
        :return: subdata
        """

        index = np.random.choice(range(self.num), self.rk, p=self.pi)
        index_ = np.array(index)
        self.subsampling = self.data[index_]
        self.subsampling_label = self.label[index_]
        return self.subsampling, self.subsampling_label

    def fit(self, train_data=None, train_label=None):

        model = self.model
        if not train_data:
            train_data = self.subsampling
            train_label = self.subsampling_label
        model.fit(train_data, train_label)
        self.model = model

    def eval(self, eval_data=[], eval_label=[]):

        model = self.model
        if eval_data == []:
            eval_data = self.data
            eval_label = self.label
        yp = model.predict(eval_data)
        acc = uts.eval(yp, eval_label, pit=False)

        return acc

    def plot_2D(self, dim1=0, dim2=1):

        plt.Figure()
        plt.scatter(self.data[:, dim1], self.data[:, dim2])
        plt.scatter(self.subsampling[:, dim1], self.subsampling[:, dim2], c="r")
        plt.show(block=True)

