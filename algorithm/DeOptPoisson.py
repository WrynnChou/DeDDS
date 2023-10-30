import matplotlib.pyplot as plt
import numpy as np
import torch
from disropt.agents import Agent
from scipy.optimize import fsolve
import Utils.Utils as uts
from algorithm.consensus_ import Consensus_
from model import get_model


class Deoptpoisson(object):
    """
    Optimal Distributed Subsampling for Maximum Quasi-Likelihood Estimators with Massive Data
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
        :param r: pre_sampling number
        :return: subdata
        """
        if r == 0:
            r = self.n
        # note that self.n is the sampling size on one client, self.n * self.size is the total size.
        r_0 = int(r)
        pi = (r_0 / self.num) * np.ones(self.num)
        index_ = []
        for i in range(self.num):
            bnenoulli_i = np.random.binomial(1, pi[i])
            if bnenoulli_i == 1:
                index_.append(i)

        index = np.array(index_)
        sampling = self.data[index, :]
        sampling_y = self.label[index]
        self.pre_pi = pi[index]
        self.pre_samp = sampling
        self.pre_samp_y = sampling_y
        self.pre_n = len(index_)

        return sampling, sampling_y

    def qse(self, method: str = "MVc"):
        """
        Get quasi-likelihood estimator from pre sampling.
        :return: \beta_{QLE},
        """

        self.method_ = method

        def gee(beta):
            """
            Rewrite the weighted estimation equation into matrix form.
            """

            a = (1 / self.pre_pi) * (self.pre_samp_y - np.exp(np.matmul(self.pre_samp, beta)))
            b = np.matmul(a, self.pre_samp)
            return b

        beta_qle = fsolve(gee, np.zeros(self.dim))
        self.beta_qle_ = beta_qle

        if method == "MVc":
            psi_ = 0
            for i in range(self.pre_n):
                psi_ += np.abs((self.pre_samp_y[i] - np.exp(np.matmul(beta_qle, self.pre_samp[i, :])))) * \
                        np.linalg.norm(self.pre_samp[i, :])
            psi = psi_ / self.pre_n
        else:
            assert method == "MV", "This method only support MV and MVc"
            psi_ = 0
            for i in range(self.pre_n):
                psi_ += np.abs((self.pre_samp_y[i] - np.exp(np.matmul(beta_qle, self.pre_samp[i, :])))) * \
                        np.linalg.norm(self.pre_samp[i, :] / self.sigma_phi(beta_qle))
            psi = psi_ / self.pre_n

        self.psi_ = psi
        return beta_qle, psi

    def sigma_phi(self, beta):
        """
        auxiliary function for MV
        """

        res_ = 0
        for i in range(self.pre_n):
            res_ += np.exp(np.matmul(beta, self.pre_samp[i, :])) * \
                    np.matmul(self.pre_samp[i, :], np.transpose(self.pre_samp[i, :])) / self.pre_pi[i]

        res = res_ / self.pre_n
        return res

    def h(self, x):

        if self.method_ == "MVc":
            res = np.linalg.norm(x)
        else:
            assert self.method_ == "MV", "This method only support MV and MVc"
            res = np.linalg.norm(x / self.sigma_phi(self.beta_qle_))

        return res

    def sampling(self, rho: float = 0.2):
        """
        Step 2
        :param rho:
        :return:
        """

        assert rho <= 1, "rho is in (0,1)"
        index = []
        for i in range(self.num):
            pi = np.minimum(self.pi_sos(self.data[i, :], self.label[i], rho), 1)
            delta_i = np.random.binomial(1, pi)
            if delta_i == 1:
                index.append(i)
        index_ = np.array(index)
        subsampling = self.data[index_, :]
        subsampling_y = self.label[index_]
        self.subsampling = subsampling
        self.subsampling_label = subsampling_y

        return subsampling, subsampling_y

    def pi_sos(self, x, y, rho):
        """
        Calculate probability of each item
        """

        pi_1 = (1 - rho) * (self.n * np.abs(y - np.exp(np.matmul(x, self.beta_qle_)))) * self.h(x) / \
               (self.num * self.psi_)
        pi_2 = rho * self.n / self.num
        pi = pi_1 + pi_2

        return pi

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