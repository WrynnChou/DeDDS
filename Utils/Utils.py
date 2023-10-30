import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import gc
import sklearn
from tqdm import tqdm
import pyunidoe as uni
from math import gcd as bltin_gcd
from scipy.special import comb, perm
from itertools import combinations, permutations
from random import sample
from sklearn.neighbors import NearestNeighbors


def coprime2(a, b):
    return bltin_gcd(a, b) == 1


def pm(a, iteration):
    '''
    1-Power iteration
    :param a:
    :param iteration:
    :return:
    '''
    dim = a.ndim
    x = np.ones((dim, 1))
    lam_prev = 0
    tol = 1e-6
    for i in range(iteration):
        x = a @ x / np.linalg.norm(a @ x)
        lam = (x.T @ a @ x) / (x.T @ x)
        if np.abs(lam - lam_prev) < tol:
            break
        lam_prev = lam
    print("The largest eigenvalue is %s and vector is %s" % (lam, x))
    return lam, x


def get_tensor(n=100, rank=100):
    """Method to create random symmetric PSD"""
    A = torch.randn(n, n)
    A = A.T @ A
    _, s, v = torch.svd(A)
    s[rank:] = 0  #
    A = v @ s.diag() @ v.T
    return A


def power_iter(mtx: torch.Tensor,
               k: int,
               shift=0.,
               t_steps: int = 10000):
    """k-Power iteration"""

    n = mtx.shape[0]  # assumes square 2D tensor
    q = torch.randn(n, k)  # initialize eigenvecs
    C = torch.as_tensor(mtx, dtype=torch.float)
    # main algo
    for _ in tqdm(range(t_steps), desc='Power iter'):
        q = C @ q - (shift * q)  # power iter w/ optional shift (explained below)
        q, _ = torch.qr(q)  # orthonormalize

    v = q  # returned eigenvec
    e = torch.stack([u.T @ C @ u for u in q.T])  # returned eigenvals
    return v, e


def glp(n, p, type='CD2'):
    '''
    Good lattice point method, to give a uniform design.
    :param n: number of points
    :param p: dimension
    :param type:  criterion
    :return: points
    '''
    if p == 1:
        design0 = np.zeros((n, p))
        for i in range(n):
            design0[i] = (2 * (i + 1) - 1) / (2 * n)
    elif p == 2 and (n + 1) in np.array((3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)):
        fb = np.array((3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987))
        H = np.array((1, fb[np.where(n <= fb)[0][0]]))
        design = np.zeros((n + 1, p))
        for i in range(p):
            for j in range(n + 1):
                design[j, i] = (2 * (j + 1) * H[i] - 1) / (2 * (n + 1)) - np.floor(
                    (2 * (j + 1) * H[i] - 1) / (2 * (n + 1)))
        design0 = design[0:n, :] * (n + 1) / (n)
    else:
        h = np.array(())
        for i in range(n - 1):
            if coprime2(i + 2, (n + 1)):
                h = np.append(h, i + 2)
        for i in range(100):
            if comb((p + i), i) > 5000:
                addnumber = i
                break
        h0 = h[sample(range(len(h)), min(len(h), (p + addnumber)))]
        H = list(combinations(h0, p))
        if len(H) > 3000:
            H_ = np.array(H)
            H_ = H_[sample(range(len(H)), 3000)]
            H = list(H_)
        design0 = np.ones((n, p))
        d0 = uni.design_eval(design0, type)
        for t in range(len(H)):
            design = np.zeros((n, p))
            for i in range(p):
                for j in range(n):
                    design[j, i] = ((j + 1) * H[t][i]) % (n + 1)
            d1 = uni.design_eval(design, type)
            if d1 < d0:
                d0 = d1
                design0 = design
        design0 = (design0 * 2 - 1) / (2 * n)
        del(H)
        del(H_)
        gc.collect()

    return design0


def diag_sign(A):
    "Compute the signs of the diagonal of matrix A"

    D = np.diag(np.sign(np.diag(A)))

    return D


def adjust_sign(Q, R):
    """
    Adjust the signs of the columns in Q and rows in R to
    impose positive diagonal of Q
    """

    D = diag_sign(Q)

    Q[:, :] = Q @ D
    R[:, :] = D @ R

    return Q, R


def IECDF_1D(sample):
    '''
    Given the inverse empirical calculated density function of 1 dimension sample
    :param sample: samples
    :return: IECDF function (or PPF)
    '''

    def iecdf(x):
        x = torch.as_tensor(x)
        index = torch.zeros_like(x)
        fix = x == 1
        n = len(sample)
        sort = sorted(sample)
        index[~fix] = torch.floor(torch.tensor(n) * x[~fix])
        index[fix] = -1 * torch.ones_like(x)[fix]
        result = np.array(sort)[index.type(torch.int)]
        return result

    return iecdf


def IECDF_nD(sample):
    """
    n-dimension IECDF
    :param sample:
    :return:
    """
    n = sample.shape[0]
    dim = sample.shape[1]
    func = list()
    for i in range(dim):
        f = IECDF_1D(sample[:, i])
        func.append(f)

    def iecdf_nd(x):
        result = np.zeros(x.shape)
        dim2 = x.shape[1]
        assert dim == dim2
        for i in range(dim2):
            result[:, i] = (func[i](x[:, i]))
        return result

    return iecdf_nd, func


def box_index(funcs, point, n=10):
    """
    It gives the index of points in the box.
    :param n: partition number, default 10
    :param funcs: IECDF_nd functions
    :param point: points
    :return: index
    """
    dim = len(funcs)
    if len(point.shape) == 1:
        num = 1
        point = point.reshape((1, point.shape[0]))
    else:
        num = point.shape[0]
    result = np.zeros((num, dim))
    box = np.zeros((n, dim))
    part = np.arange(0, 1, (1 / n))
    for i in range(dim):
        for j in range(n):
            box[j, i] = funcs[i](part[j])
    for i in range(dim):
        for j in range(num):
            ind_lower = np.where(box[:, i] <= point[j, i])
            result[j, i] = max(ind_lower[0])
    return result


def data_index(index, data, block):
    if len(index.shape) == 1:
        n = 1
        dim = index.shape[0]
        index = index.reshape((1, index.shape[0]))
    else:
        n = index.shape[0]
        dim = index.shape[1]
    num = data.shape[0]
    result = list()
    I = list()
    for i in range(n):
        p_index = torch.ones(num).bool()
        for j in range(dim):
            f = IECDF_1D(data[:, j])
            ind = index[i, j]
            lower = f(((ind) / block))
            upper = f(((ind + 1) / block))
            l_index = data[:, j] >= lower
            u_index = data[:, j] <= upper
            p_index = p_index & (l_index & u_index)
        result.append(data[np.where(p_index == True)[0], :])
        I.append(np.where(p_index == True)[0])
    return result, I


def nus_nnbrs(data, iecdf, funcs, UD, block):
    """
    Non-uniform stratification nearest neighbors
    :param data
    :param funcs: result of IECDF_nD
    :param UD: the points you want find neighbors
    :return: Nearest neighbor
    """
    dim = UD.shape[1]
    UD_inv = iecdf(UD)
    idx_UD = box_index(funcs, UD_inv, block)
    neighbor_data, I = data_index(idx_UD, data, block)
    nbr_points = np.zeros_like(idx_UD)
    nbr_i = list()
    for i in range(len(neighbor_data)):
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(neighbor_data[i])
        dis, indx = neigh.kneighbors(UD_inv[i].reshape(1, 2))
        nbr_points[i, :] = neighbor_data[i][indx]
        nbr_i.append(I[i][indx])
    return nbr_points, np.array(nbr_i).reshape(len(nbr_i))


def eval(y_predict, y_ture, pit = True):
    """
    Eval the percentage of y == yp
    """
    d1 = y_predict.shape[0]
    d2 = y_ture.shape[0]
    assert d1 == d2, "the numbers of two ys don`t match. Please check the number."
    acc = (y_ture == y_predict).sum() / len(y_predict)
    if pit == True:
        print("The accuracy is %.2f%%." % (acc * 100))
    return np.around((acc * 100), 2)


def fsl(n, m, p, seed: int = None):
    """
    flexible sliced LHD
    :param n: number of points
    :param m: number of slide, array in decreasing order, like [5, 3, 1]
    :param p: number of dimensions
    :return: fsl
    """
    assert len(m) != 1, print('S = 1, no sliced!')
    n1 = sum(m)
    assert n == n1, print("m doesn't match n!")
    s = m.shape[0]
    N = np.zeros((1, n))
    if seed != None:
        np.random.seed(seed)
    rng = np.random.default_rng()
    for i in range(p):
        per = rng.permutation(n).reshape((1, n))
        N = np.concatenate((N, per))
    N_ = N[1:(p + 1)]
    result = np.zeros((p, 1))
    for i in range(s):
        l, N_ = divide(m[i], N_)
        result = np.concatenate((result, l), 1)
    result = result[:, 1:(n + 1)]
    return result


def divide(mi, N_):
    """
    function used in fsl, divide the number into block.
    """
    n = N_.shape[1]
    p = N_.shape[0]
    ti = np.floor((n / mi))
    l = np.zeros((p, mi))
    mask = np.zeros_like(N_)
    for i in range(mi):
        r = np.random.randint(0, ti, (p, 1))
        l[:, i] = N_[np.arange(0, p), (ti * i + r).reshape(p).astype(int)]
        mask[np.arange(0, p), (ti * i + r).reshape(p).astype(int)] = np.ones(p)
    res = N_[~(mask.astype(bool))].reshape((p, (n - mi)))
    return l, res


def logit(x):
    """
    Logit function
    """
    ones = np.ones_like(x)
    y = ones / (ones + math.e ** x)
    return y


def classify(y):
    """
    if y > 0.5, then return 1, else 0.
    r is the bool mask.
    """
    r = y > 0.5
    result = r.astype('int')
    return result, r


def train_logistics(train, train_label, test, test_label):
    shape = train.shape
    shape2 = test.shape

    if len(shape) == 1:
        assert len(shape2) == 1, "the dimensions of train and test don't match"
        n1 = shape
        t1 = shape2
    else:
        n1 = shape[0]
        t1 = shape2[0]

    n2 = train_label.shape[0]
    t2 = test_label.shape[0]

    assert n1 == n2, "Numbers of train data and label don't match"
    assert t1 == t2, "Numbers of test data and label don't match"

    model = sklearn.linear_model.LogisticRegression()
    model.fit(train, train_label)
    model.classes_
    yp = model.predict(test)
    acc = eval(yp, test_label)

    return model, acc


def sigma(size, num):
    result = np.zeros((size, size))
    for i in range(size):

        for j in range(size):
            result[i, j] = num ** (np.abs(i - j))

    return result


def find_second_largest(x: np.ndarray):
    """
    find the second-largest element of x
    """
    n = len(x)
    first = -np.inf
    second = -np.inf
    assert n > 1, "x only has one element!"
    for i in x:
        if i > first:
            second = first
            first = i
        elif (i > second) & (i <= first):
            second = i

    return second


def divi(fsl: np.ndarray, m):
    """
    Divide the slice Ud and reshape it to broadcast.
    :param fsl: fsl output (p, n * m)
    :param m: block
    :return: np.array with shape (m, n ,p)
    """

    nm = fsl.shape[1]
    assert (nm / m).is_integer() == True, "fsl can't be divide by m"
    n = int(nm / m)
    p = fsl.shape[0]
    res = np.zeros((m, n, p))
    fsl_ = fsl.transpose()
    for i in range(m):
        res[i, :, :] = fsl_[i * n: (i + 1) * n, :]
    res_ = res / res.max()
    return res_


def kernel(u: np.ndarray, v: np.ndarray, criterion: str = "CD"):
    """
    Kernel function
    """

    res = 0
    d1 = u.shape[0]
    d2 = v.shape[0]
    assert d1 == d2, "u, v have different dimensions"
    if criterion == "CD":
        for i in range(d1):
            res += 1 + 1 / 2 * np.abs(u[i] - 1 / 2) + 1 / 2 * np.abs(v[i] - 1 / 2) - 1 / 2 * np.abs(u[i] - v[i])

    elif criterion == "WD":
        for i in range(d1):
            res += 3 / 2 - np.abs(u[i] - v[i]) + (u[i] - v[i]) ** 2

    else:
        assert criterion == "MD", "Only support CD, WD and MD."
        for i in range(d1):
            res += 15 / 8 - 1 / 4 * np.abs(u[i] - 1 / 2) - 1 / 4 * np.abs(v[i] - 1 / 2) - 3 / 4 * np.abs(u[i] - v[i]) + \
                   1 / 2 * (u[i] - v[i]) ** 2
    return res


def ecdf_1dimension(x: np.ndarray):
    n = x.shape[0]
    assert len(x.shape) == 1, "This function only support one dimension"

    def ecdf(u: np.ndarray):
        u = np.array(u)
        if len(u.shape) == 0:
            u = u.reshape(1)
        res = np.zeros_like(u)
        for i in range(u.shape[0]):
            s = np.sum(x <= u[i])
            res[i] = s / n
        return res

    return ecdf


def Tx(x: np.ndarray):
    """
    Generate transformation T_{\mathcal{x}}
    :param x: whole samples
    :return: transformation function Tx
    """

    dim = x.shape[1]
    f_ = []
    for i in range(dim):
        f = ecdf_1dimension(x[:, i])
        f_.append(f)

    def T(u):
        res = []
        for i in range(dim):
            res_ = f_[i](u[i])
            res.append(res_)
        res = np.array(res)
        return res

    return T


def gefd(data: np.ndarray, subdata: np.ndarray):
    """
    General empirical F-discrepancy
    :param data: Whole data
    :param subdata: sub data
    :return: FEFD of subdata.
    """

    dim1 = data.shape[1]
    dim2 = subdata.shape[1]
    assert dim1 == dim2, "Dimensions don't please check the data."
    N = data.shape[0]
    n = subdata.shape[0]
    T = Tx(data)

    d1 = 0
    for i in range(N):
        for j in range(N):
            Tu = T(data[i, :])
            Tv = T(data[j, :])
            d1 += kernel(Tu, Tv)

    d2 = 0
    for i in range(N):
        for j in range(n):
            Tu = T(data[i, :])
            Tv = T(subdata[j, :])
            d2 += kernel(Tu, Tv)

    d3 = 0
    for i in range(n):
        for j in range(n):
            Tu = T(subdata[i, :])
            Tv = T(subdata[j, :])
            d3 += kernel(Tu, Tv)

    res = 1 / (N ** 2) * d1 - 2 / (N * n) * d2 + 1 / (n ** 2) * d3
    return res


def fast_gefd(data, subdata):
    """
    Fast General empirical F-discrepancy compare method.
    For the same data, we want to compare the GEFD of two subdata, they share the same d1.
    And d1 is the most time-cost part. Thus, we could only calculate d2 and d3 to save time.
    :param data: The whole data
    :param subdata: subdata
    :return: The sum of second part and third part of GEFD between data and subdata.
    """

    dim1 = data.shape[1]
    dim2 = subdata.shape[1]
    assert dim1 == dim2, "Dimensions don't please check the data."
    N = data.shape[0]
    n = subdata.shape[0]
    T = Tx(data)

    d2 = 0
    for i in range(N):
        for j in range(n):
            Tu = T(data[i, :])
            Tv = T(subdata[j, :])
            d2 += kernel(Tu, Tv)

    d3 = 0
    for i in range(n):
        for j in range(n):
            Tu = T(subdata[i, :])
            Tv = T(subdata[j, :])
            d3 += kernel(Tu, Tv)

    print(d2)
    print(d3)
    res = - 2 / (N * n) * d2 + 1 / (n ** 2) * d3
    return res

def star_graph(N: int, center: int=0) -> np.ndarray:
    """
    Construct a star graph
    :param N: number of agents
    :return: graph
    """

    Adj = np.zeros(([N, N]))
    for i in range(N):
        Adj[i, center] = 1
        Adj[center, i] = 1
    Adj[center, center] = 0

    return Adj

def hist_iecdf_1d(hist: np.ndarray, bins: np.ndarray):
    """
    Get iecdf from histogram. Generate samples of histogram then use these sample on IECDF_1d
    :param hist:
    :param bins:
    :return:
    """
    def f(x):
        max_ = np.max(x)
        min_ = np.min(x)
        assert max_ <= 1, "IECDF input should be in (0, 1)."
        assert min_ >= 0, "IECDF input should be in (0, 1)."

        x = np.array(x)
        if len(x.shape) == 0:
            x = x.reshape(1)
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            p_ = np.where(hist <= x[i])
            idx = p_[0][-1].astype("int")
            y[i] = bins[idx]
        return y

    return f

def hist_iecdf_nd(hist: list, bins: list):
    """
    nd hist_iecdf
    :param hist:
    :param bins:
    :return:
    """

    d = len(hist)
    funcs = list()
    for i in range(d):
        f = hist_iecdf_1d(hist[i], bins[i])
        funcs.append(f)

    def iecdf_nd(x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, d)
        y = np.zeros_like(x)
        for i in range(d):
            y[:, i] = funcs[i](x[:, i])
        return y

    return iecdf_nd, funcs

def two_points(hist: np.ndarray, bins: np.ndarray):
    """
    inverse function of two points line function
    :param hist: values
    :param bins: positions
    :return: liner function
    """

    assert len(hist) == 2, "This functions only support two points"
    assert bins[0] < bins[1], "The bins gets wrong"
    def g(x):
        k = (hist[1] - hist[0]) / (bins[1] - bins[0])
        y = k * (x - bins[0]) + hist[0]
        return y
    def f(x):
        x = np.array(x)
        if len(x.shape) == 0:
            p1 = x >= bins[0]
            p2 = x < bins[1]
            p = p1 & p2
            if p == True:
                y = g(x)
            else:
                y = 0
        else:
            y = np.zeros_like(x)
            p1 = x >= bins[0]
            p2 = x < bins[1]
            p = p1 & p2
            for i in range(np.max(y.shape[0])):
                if p[i] == True:
                    y[i] = g(x[i])
                else:
                    y[i] = 0
        return y

    return f

def liner_inv_1d(hist: np.ndarray, bins: np.ndarray, num = 100):

    """
    liner inverse 1d function by histogram.
    This function firstly estimate ECDF by liner histogram.
    :param num: number of percentile.
    :return: IECDF_1D function
    """

    assert len(bins.shape) == 1, " Wrong dimension"
    n = bins.shape[0]

    def f(x):
        f_list = []
        for i in range((n-1)):
            f_ = two_points(hist[i:(i + 2)], bins[i:(i + 2)])
            f_list.append(f_(x))
        y_ = np.array(f_list)
        y = np.sum(y_, axis=0)
        return y

    def g(x):
        x = np.array(x)
        if len(x.shape) == 0:
            x = x.reshape(1)
        p = np.zeros_like(x)
        x_range = np.linspace(bins[0], bins[-1], num)
        y = f(x_range)
        for i in range(x.shape[0]):
            p_ = np.where(y <= x[i])
            p[i] = p_[0][-2]
        result = x_range[(p.astype('int'))]
        return result

    return g

def liner_inv_nd(hist: list, bins: list, num = 100):
    """nd liner inverse function """

    dim = len(hist)
    func = list()
    for i in range(dim):
        f = liner_inv_1d(hist[i], bins[i], num)
        func.append(f)

    def fn(x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, dim)
        result = np.zeros_like(x)
        dim2 = x.shape[1]
        assert dim == dim2
        for i in range(dim2):
            result[:, i] = (func[i](x[:, i]))
        return result

    return fn, func

def write_glp(n: int, p: int, type: str="CD"):
    """
    This function is used to write glp in order to save memory.
    """

    m = glp(n, p, type)
    np.savetxt("./temp/temp_glp.txt", m)






