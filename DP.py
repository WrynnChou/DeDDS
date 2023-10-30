import numpy as np
import math
from algorithm.DP import *
import json
import matplotlib.pyplot as plt
import sklearn.datasets
from Utils.Utils import *
from Utils.Utils import ecdf_1dimension
from algorithm.DDS import Dds

x = np.random.multivariate_normal(np.array([0,0]),np.eye(2) , 1000)
f1 = ecdf_1dimension(x[:, 0])
f2 = ecdf_1dimension(x[:, 1])

xx = np.linspace(-5, 5, 1000)
yy1 = np.zeros_like(xx)
for i in range(1000):
    yy1[i] = f1(xx[i])
yy2 = np.zeros_like(xx)
for i in range(1000):
    yy2[i] = f2(xx[i])
delta = 0.1
epsion = 1
sigma = 2 * math.log(1.25 / delta) / (epsion ** 2)

range_ = np.array([np.min(x), np.max(x)])
hist_list, bins_list = [],[]
for i in range(2):
    hist, bins = hist_est(x[:, i], range_, num=20, plot=True)
    hist_list.append(hist)
    bins_list.append(bins)

fff, funcs = liner_inv_nd(hist_list, bins_list)
fff2, funcs2 = hist_iecdf_nd(hist_list, bins_list)

gg = IECDF_1D(x[:, 1])
xxx = np.linspace(0, 1, 100)
y2 = funcs2[1](xxx)

y1 = gg(xxx)
plt.figure()
plt.scatter(xxx, y1)
plt.scatter(xxx, y2, c = "r")
plt.show(block = True)



xxxx, yyyy = sklearn.datasets.make_moons(10000, noise=0.1)
xxxx1 = xxxx[yyyy == 1, :]

ud = glp(100, 2)
with open('Utils/conf3.json', 'r') as f:
    conf = json.load(f)
dds = Dds(conf, xxxx, yyyy)
dds.pca()




print("Good Luck！")