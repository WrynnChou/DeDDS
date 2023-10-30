import numpy as np
import matplotlib.pyplot as plt
import torch
import Utils.Utils as uts
from sklearn.decomposition import PCA
from model import get_model
from sklearn.neighbors import NearestNeighbors
class Dds(object):
    """
    Model-free Subsampling Method Based on Uniform Designs
    """

    def __init__(self, conf, data, label):

        self.conf = conf

        self.n = conf['subsampling_number']

        self.method = conf['method']

        self.Explained_variance_ratio_threshold = conf['Explained_variance_ratio_threshold']

        self.data = data

        self.label = label

        self.dim = self.data.shape[1]

        self.num = self.data.shape[0]

        self.model = get_model(self.conf['method'])


    def pca(self):

        pca = PCA(n_components=self.Explained_variance_ratio_threshold)
        pc = pca.fit_transform(self.data)
        self.n_components = pca.n_components_
        self.pc = pc

    def sampling(self, ud, block = 5):

        prin = self.pc
        dim = self.n_components
        dim2 = ud.shape[1]
        assert dim == dim2, print("The dimensions don't match!")

        iecdf, funcs = self.IECDF_nD(prin)
        sub_prin, sub_idx = self.nus_nnbrs(prin, iecdf, funcs, ud, block)
        sampling = self.data[sub_idx, :]
        sampling_label = self.label[sub_idx]
        self.sub_prin = sub_prin
        self.subsampling = sampling
        self.subsampling_y = sampling_label

        return sampling, sampling_label

    def IECDF_1D(self, sample):
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

    def IECDF_nD(self, sample):
        '''
        n-dimension IECDF
        :param sample:
        :return:
        '''
        n = sample.shape[0]
        dim = sample.shape[1]
        func = list()
        for i in range(dim):
            f = self.IECDF_1D(sample[:, i])
            func.append(f)

        def iecdf_nd(x):
            result = np.zeros(x.shape)
            dim2 = x.shape[1]
            assert dim == dim2
            for i in range(dim2):
                result[:, i] = (func[i](x[:, i]))
            return result

        return iecdf_nd, func


    def nus_nnbrs(self, data, iecdf, funcs, UD, block):
        '''
        Non-uniform stratification nearest neighbors
        :param data
        :param funcs: result of IECDF_nD
        :param UD: the points you want find neighbors
        :return: Nearest neighbor
        '''
        dim = UD.shape[1]
        UD_inv = iecdf(UD)
        idx_UD = self.box_index(funcs, UD_inv, block)
        neighbor_data, I = self.data_index(idx_UD, data, block)
        nbr_points = np.zeros_like(idx_UD)
        nbr_i = list()
        for i in range(len(neighbor_data)):

            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(neighbor_data[i])
            dis, indx = neigh.kneighbors(UD_inv[i].reshape(1, dim))
            nbr_points[i, :] = neighbor_data[i][indx]
            nbr_i.append(I[i][indx])

        return nbr_points, np.array(nbr_i).reshape(len(nbr_i))

    def box_index(self, funcs, point, n=10):
        '''
        It gives the index of points in the box.
        :param n: partition number, default 10
        :param funcs: IECDF_nd functions
        :param point: points
        :return: index
        '''
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

    def data_index(self,index, data, block):
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
                f = self.IECDF_1D(data[:, j])
                ind = index[i, j]
                lower = f(((ind) / block))
                upper = f(((ind + 1) / block))
                l_index = data[:, j] >= lower
                u_index = data[:, j] <= upper
                p_index = p_index & (l_index & u_index)
            result.append(data[np.where(p_index == True)[0], :])
            I.append(np.where(p_index == True)[0])
        return result, I

    def fit(self, train_data=[], train_label=[]):

        model = self.model
        if train_data == []:
            train_data = self.subsampling
            train_label = self.subsampling_y
        model.fit(train_data, train_label)
        self.model = model
    def eval(self, eval_data=[], eval_label=[], pit = True):

        model = self.model
        if eval_data == []:
            eval_data = self.data
            eval_label = self.label
        yp = model.predict(eval_data)
        acc = uts.eval(yp, eval_label, pit)

        return acc

    def plot_2D(self, dim1 = 0, dim2 = 1 ):

        plt.Figure()
        plt.scatter(self.data[:, dim1], self.data[:, dim2])
        plt.scatter(self.subsampling[:, dim1], self.subsampling[:, dim2], c="r")
        plt.show(block=True)

    def sampling_dpconsesus(self, ud: np.ndarray, iecdf, funcs, block: int = 5):
        """
        Sampling with consensus ECDF

        """

        dim = self.n_components
        dim2 = ud.shape[1]
        assert dim == dim2, print("The dimensions don't match!")
        prin = self.pc
        sub_prin, sub_idx = self.nus_nnbrs(prin, iecdf, funcs, ud, block)
        subsampling = self.data[sub_idx, :]
        subsampling_label = self.label[sub_idx]
        self.subsampling = subsampling
        self.subsampling_y = subsampling_label

        return subsampling, subsampling_label