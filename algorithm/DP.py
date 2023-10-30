import numpy as np
import math
import time
import matplotlib.pyplot as plt
from Utils.Utils import ecdf_1dimension



def adapt_quantile(data: np.ndarray, range_data: np.ndarray, num: int = 10, epsilon: np.float64 = 0.1, delta: np.float64 = 1
                   , kind: str = 'Gaussian', log_plot = False, plot = True):

    """
    Estimate ECDF with DP. Using quantile to dichotomy the range and find the biggest field to query
    the hist upper and lower.
    :param epsilon: DP parameter
    :param delta: DP parameter
    :param kind: mechanism
    :return: hist
    """

    assert len(data.shape) == 1, "This function only support 1 dimension data!"
    assert range_data.shape[0] == 2, "Range should be two elements array"
    assert range_data[0] < range_data[1], "Upper should be larger than lower"

    if kind == 'Gaussian':
        sigma = 2 * math.log(1.25 / delta) / (epsilon ** 2)
        method = np.random.normal
    else:
        assert kind == 'Laplace', "This function only support gaussian and laplace mechanism"
        sigma = 1 / epsilon
        method = np.random.laplace

    f = ecdf_1dimension(data)
    quantile = np.array([0, 1])
    value = np.array([range_data[0], range_data[1]])
    x_plt = np.linspace(range_data[0], range_data[1], 1000)
    y_plt = batch_ecdf(f, x_plt)

    for i in range(num):

        gap = quantile - np.roll(quantile, 1)
        idx = np.argmax(gap)
        next_val = 0.5 * (value[idx] + value[(idx - 1)])
        upper = sum(data >= next_val)
        lower = sum(data < next_val)

        upper_dp = np.around(upper + method(0, sigma, 1))
        lower_dp = np.around(lower + method(0, sigma, 1))
        next_quan = lower_dp / (lower_dp + upper_dp)

        value = np.sort(np.append(value, [next_val]))
        quantile = np.sort(np.append(quantile, next_quan))
        if log_plot == True:
            plt.figure()
            plt.plot(x_plt, y_plt)
            plt.plot(value, quantile, c="orange")
            plt.scatter(value, quantile, c="orange")
            plt.savefig("figures/dapt%s.eps" % (i), format = "eps")
            plt.close()
    if plot == True:
        plt.figure()
        plt.plot(x_plt, y_plt)
        plt.plot(value, quantile, c="orange")
        plt.title(r"Histogram estimation by %s mechanism with $\epsilon=%s, \delta=%s$" % (kind, epsilon, delta))
        plt.show(block = True)

    return value, quantile

def hist_est(data: np.ndarray, range_data: np.ndarray, num: int = 10, epsilon: np.float64 = 0.1, delta: np.float64 = 1
                   , kind: str = 'Gaussian', plot = True, density: bool = True, seed: int = 0):

    assert len(data.shape) == 1, "This function only support 1 dimension data!"
    assert range_data.shape[0] == 2, "Range should be two elements array"
    assert  range_data[0] < range_data[1], "Upper should be larger than lower"

    if kind == 'Gaussian':
        sigma = 2 * math.log(1.25 / delta) / (epsilon ** 2)
        method = np.random.normal
    else:
        assert kind == 'Laplace', "This function only support gaussian and laplace mechanism"
        sigma = 1 / epsilon
        method = np.random.laplace

    f = ecdf_1dimension(data)
    x_plt = np.linspace(range_data[0], range_data[1], 1000)
    y_plt = batch_ecdf(f, x_plt)

    hist, bins = np.histogram(data, num, range=range_data)
    if seed != 0:
        np.random.seed(seed)
    hist_dp = np.floor(hist + method(0, sigma, num))
    for i in range(len(hist_dp)):
        if hist_dp[i] <= 0:
            hist_dp[i] = 0
    # Original hist may be 0 and add noise might make it less than 0. It is unreasonable.
    # Thus we use 0 taking the place of negative value。
    dens_dp = hist_dp / sum(hist_dp)
    cdf_dp_ = np.zeros_like(dens_dp)
    a = 0
    for i in range(len(hist_dp)):
        cdf_dp_[i] = a
        a += dens_dp[i]
    point_ = mid(bins)
    point = np.insert(point_, [0, num], [bins[0], bins[-1]])
    cdf_dp = np.insert(cdf_dp_, [0, num], [0, 1])
    hist_dp = np.insert(hist_dp, [0, num], [0, 0])

    if plot == True:
        plt.figure()
        plt.plot(x_plt, y_plt)
        plt.plot(point, cdf_dp, c="orange")
        plt.title(r"Histogram estimation by %s mechanism with $\epsilon = %s, \delta = %s$" % (kind, epsilon, delta))
        plt.show(block=True)

    if density == True:
        return cdf_dp, point
    else:
        return hist_dp, point

def direct_dp(data: np.ndarray, range_data: np.ndarray, epsilon: np.float64 = 0.1, delta: np.float64 = 1,
              kind: str='Gaussian', plot = True):

    """
    Directly add noise to data. ##### This method perform extremely bad ######
    Because it directly show the data and for DP the noise is much larger.
    """

    assert len(data.shape) == 1, "This function only support 1 dimension data!"
    assert range_data.shape[0] == 2, "Range should be two elements array"
    assert range_data[0] < range_data[1], "Upper should be larger than lower"

    if kind == 'Gaussian':
        sigma = 2 * math.log(1.25 / delta) / (epsilon ** 2)
        method = np.random.normal
    else:
        assert kind == 'Laplace', "This function only support gaussian and laplace mechanism"
        sigma = 1 / epsilon
        method = np.random.laplace

    n = data.shape[0]
    data_dp = data + method(0, sigma, n)

    f = ecdf_1dimension(data)
    f_dp = ecdf_1dimension(data_dp)
    x_plt = np.linspace(range_data[0], range_data[1], 1000)
    y = batch_ecdf(f, x_plt)
    y_dp = batch_ecdf(f_dp, x_plt)
    if plot == True:
        plt.figure()
        plt.plot(x_plt, y)
        plt.plot(x_plt, y_dp, c="orange")
        plt.title(r"Naive method by %s mechanism with $\epsilon = %s, \delta = %s$" % (kind, epsilon, delta))
        plt.show(block=True)
    return y_dp

def batch_ecdf(f, x: np.ndarray):
    """
    The original ecdf do not support batch. This function improve it.
    :param f: ecdf
    :param x: plt points
    :return: y plot
    """

    y_plt = np.zeros_like(x)
    n = x.shape[0]
    for i in range(n):
        y_plt[i] = f(x[i])

    return y_plt

def mid(x:np.ndarray):
    n = x.shape[0]
    m = n - 1
    out = np.zeros(m)
    for i in range(m):
        out[i] = 0.5 * (x[i] + x[(i + 1)])
    return out

def hist_to_ecdf(hist_list: list):

    dim = len(hist_list)
    out_list = []
    for i in range(dim):
        n_ = np.sum(hist_list[i])
        out_ = np.zeros_like(hist_list[i])
        s = 0
        for j in range(hist_list[i].shape[0]):
            s += hist_list[i][j]
            out_[j] = s
        out_ = out_ / n_
        out_list.append(out_)

    return out_list