
import numpy as np
import  matplotlib.pyplot as plt
from Utils.Utils import eval
from model import get_model
class Urs(object):

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

    def sampling(self):

        index = np.random.randint(0, self.num, self.n)
        sampling = self.data[index, :]
        sampling_y = self.label[index]
        self.subsampling = sampling
        self.subsampling_y = sampling_y

        return sampling, sampling_y

    def fit(self, train_data = [], train_label = []):

        model = self.model
        if train_data == []:
            train_data = self.subsampling
            train_label = self.subsampling_y
        model.fit(train_data, train_label)
        self.model = model

    def eval(self, eval_data=[], eval_label=[], pit=True):

        model = self.model
        if eval_data == []:
            eval_data = self.data
            eval_label = self.label
        yp = model.predict(eval_data)
        acc = eval(yp, eval_label, pit=pit)

        return acc

    def plot_2D(self, dim1 = 0, dim2 = 1 ):

        plt.Figure()
        plt.scatter(self.data[:, dim1], self.data[:, dim2])
        plt.scatter(self.subsampling[:, dim1], self.subsampling[:, dim2], c="r")
        plt.show(block=True)