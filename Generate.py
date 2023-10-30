import numpy as np
import json
import sklearn
import matplotlib.pyplot as plt
import Utils.Utils as uts
from algorithm.SP import Sp
import logging
import gc

def train_eval(cla, m, conf2, conf, train_list, label_list, test_list, test_label_list):
    """
    Use all and distributed method sampling.
    :param cla: method
    :return: accurate
    """

    assert (m + 1) == len(train_list), "m doesn't fitted!"
    cla_list = []
    logging.disable(logging.CRITICAL)

    for i in range((m + 1)):

        if i == 0:

            cla_ = cla(conf2, train_list[i], label_list[i])

        else:

            cla_ = cla(conf, train_list[i], label_list[i])

        cla_list.append(cla_)

    acc = np.zeros(((m + 1), (m + 2)))
    sampling_list = []
    sampling_label_list = []

    j = 0
    for i in cla_list:

        if cla == Sp:

            i.sampling()
            x, y = i.nnbrs()

        elif cla == Dds:

            i.pca()
            p = i.n_components
            ud = uts.glp(i.n, p)
            x, y = i.sampling(ud)

        else:

            x, y = i.sampling()

        if len(sampling_list) == 0:

            sampling_list = x
            sampling_label_list = y

        else:

            sampling_list = np.concatenate([sampling_list, x])
            sampling_label_list = np.concatenate([sampling_label_list, y])

        i.fit()
        # fit on local data
        # eval on different data
        for k in range((m + 1)):
            acc[k, j] = i.eval(test_list[k], test_label_list[k])

        j = j + 1

    # train on all the sampling
    model = sklearn.linear_model.LogisticRegression()
    model.fit(sampling_list, sampling_label_list)

    for k in range((m + 1)):
        acc[k, (m + 1)] = uts.eval(model.predict(test_list[k]), test_label_list[k])

    logging.disable(logging.NOTSET)
    return acc

#
mu1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
mu2 = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
mu3 = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
mu4 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
<<<<<<< HEAD
sigma = uts.sigma(7, 0.5)
beta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape((7, 1))

np.random.seed(97)
x1 = np.random.multivariate_normal(mu1, sigma, 10250)
x2 = np.random.multivariate_normal(mu2, sigma, 10250)
x3 = np.random.multivariate_normal(mu3, sigma, 10250)
x4 = np.random.multivariate_normal(mu4, sigma, 10250)
=======
# mu1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# mu2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# mu3 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
# mu4 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
sigma1 = uts.sigma(7, 0.5)
sigma2 = uts.sigma(7, 0)
sigma3 = uts.sigma(7, 0.2)
sigma4 = uts.sigma(7, 0.7)
beta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape((7, 1))

np.random.seed(97)
x1 = np.random.multivariate_normal(mu1, sigma1, 10250)
x2 = np.random.multivariate_normal(mu2, sigma1, 10250)
x3 = np.random.multivariate_normal(mu3, sigma1, 10250)
x4 = np.random.multivariate_normal(mu4, sigma1, 10250)


# x1 = np.random.multivariate_normal(mu1, sigma, 10250)
# x2 = np.random.multivariate_normal(mu1, sigma1, 10250)
# x3 = np.random.multivariate_normal(mu1, sigma2, 10250)
# x4 = np.random.multivariate_normal(mu1, sigma3, 10250)
#
x1 = multivariate_t_rvs(mu1, sigma1, 5, 10250)
x2 = multivariate_t_rvs(mu2, sigma2, 5, 10250)
x3 = multivariate_t_rvs(mu3, sigma3, 5, 10250)
x4 = multivariate_t_rvs(mu4, sigma4, 5, 10250)

>>>>>>> 461b612 (README)
#
train1 = x1[0:10000, :]
train2 = x2[0:10000, :]
train3 = x3[0:10000, :]
train4 = x4[0:10000, :]
test1  = x1[10000:10250, :]
test2  = x2[10000:10250, :]
test3  = x3[10000:10250, :]
test4  = x4[10000:10250, :]
x = np.concatenate([train1, train2, train3, train4])
x_test = np.concatenate([test1, test2, test3, test4])
#
np.random.seed()
<<<<<<< HEAD
y1 = uts.logit(np.matmul(x1, beta).ravel())
label1, mask1 = uts.classify(y1)
y2 = uts.logit(np.matmul(x2, beta).ravel())
label2, mask2 = uts.classify(y2)
y3 = uts.logit(np.matmul(x3, beta).ravel())
label3, mask3 = uts.classify(y3)
y4 = uts.logit(np.matmul(x4, beta).ravel())
=======
y1 = uts.logit(np.matmul(x1, beta).ravel() + np.random.normal(0, 1, 10250))
label1, mask1 = uts.classify(y1)
y2 = uts.logit(np.matmul(x2, beta).ravel() + np.random.normal(0, 1, 10250))
label2, mask2 = uts.classify(y2)
y3 = uts.logit(np.matmul(x3, beta).ravel() + np.random.normal(0, 1, 10250))
label3, mask3 = uts.classify(y3)
y4 = uts.logit(np.matmul(x4, beta).ravel() + np.random.normal(0, 1, 10250))
>>>>>>> 461b612 (README)
label4, mask4 = uts.classify(y4)

train_label1 = label1[0:10000]
train_label2 = label2[0:10000]
train_label3 = label3[0:10000]
train_label4 = label4[0:10000]
test_label1  = label1[10000:10250]
test_label2  = label2[10000:10250]
test_label3  = label3[10000:10250]
test_label4  = label4[10000:10250]
#
#
label = np.concatenate([train_label1, train_label2, train_label3, train_label4])
label_test = np.concatenate([test_label1, test_label2, test_label3, test_label4])

np.savetxt("./data/x_simulation1.txt", x)
np.savetxt("./data/label_simulation1.txt", label)
np.savetxt("./data/x_simulation_test1.txt", x_test)
np.savetxt("./data/label_simulation_test1.txt", label_test)
#
# train_label1 = label1[0:9500]
# train_label2 = label2[0:9500]
# train_label3 = label3[0:9500]
# test_label1  = label1[9501:10000]
# test_label2  = label2[9501:10000]
# test_label3  = label3[9501:10000]
# train_y1 = y1[0:9500]
# train_y2 = y2[0:9500]
# train_y3 = y3[0:9500]
# test_y1  = label1[9501:10000]
# test_y2  = label2[9501:10000]
# test_y3  = label3[9501:10000]
#
# x = np.concatenate([x1, x2, x3])
# label = np.concatenate([label1, label2, label3])
# train = np.concatenate([train1, train2, train3])
# train_label = np.concatenate([train_label1, train_label2, train_label3])
# train_y = np.concatenate([train_y1, train_y2, train_y3])
# test = np.concatenate([test1, test2, test3])
# test_label = np.concatenate([test_label1, test_label2, test_label3])
# test_y = np.concatenate([test_y1, test_y2, test_y3])
#
# #all data
# model_all, acc_all = uts.train_logistics(train, train_label, test, test_label)
# acc_all_1 = uts.eval(model_all.predict(test1), test_label1)
# acc_all_2 = uts.eval(model_all.predict(test2), test_label2)
# acc_all_3 = uts.eval(model_all.predict(test3), test_label3)
#
#
# with open('Utils/conf.json', 'r') as f:
#     conf = json.load(f)
# with open('Utils/conf2.json', 'r') as f:
#     conf2 = json.load(f)
#
#
# train_list = [train, train1, train2, train3]
# label_list = [label, label1, label2, label3]
# test_list = [test, test1, test2, test3]
# test_label_list = [test_label, test_label1, test_label2, test_label3]
#
#
#
# acc_sp = train_eval(Sp, 3, conf2, conf, train_list, label_list, test_list, test_label_list)
# acc_urs = train_eval(Urs, 3, conf2, conf, train_list, label_list, test_list, test_label_list)
# acc_iboss = train_eval(Iboss, 3, conf2, conf, train_list, label_list, test_list, test_label_list)
# acc_dds = train_eval(Dds, 3, conf2, conf, train_list, label_list, test_list, test_label_list)
#
#
# plt.Figure()
# plt.scatter(x1[:, 0], x1[:, 1])
# plt.scatter(x2[:, 0], x2[:, 1], c="r")
# plt.scatter(x3[:, 0], x3[:, 1], c="y")
# plt.show(block=True)
#
# plt.Figure()
# plt.scatter(x1[mask1, 0], x1[mask1, 1])
# plt.scatter(x1[~mask1, 0], x2[~mask1, 1], c="r")
# plt.show(block=True)
#
# print('ss')

# ------------ Simulation 2 --------------------------

# np.random.seed(777)
# mu1 = np.array([0, 0, 0])
# mu2 = np.array([2, 2, 2])
# beta = np.array([0.5, 0.5, 0.5])
# sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# x1 = np.random.multivariate_normal(mu1, sigma, 1000)
# x2 = np.random.multivariate_normal(mu2, sigma, 1000)
# y1 = np.matmul(x1, beta)
# y2 = np.matmul(x2, beta)
# np.savetxt("./data/simulation2_x1.txt", x1)
# np.savetxt("./data/simulation2_x2.txt", x2)
from algorithm.DDS import Dds
import json
x1 = np.loadtxt("./data/simulation2_x1.txt")
x2 = np.loadtxt("./data/simulation2_x2.txt")
beta = np.array([0.5, 0.5, 0.5])
y1 = np.matmul(x1, beta)
y2 = np.matmul(x2, beta)
x1_s = np.loadtxt("./data/simulation2_x1_s.txt")
x2_s = np.loadtxt("./data/simulation2_x2_s.txt")
x3_s = np.loadtxt("./data/simulation2_x3_s.txt")
x_s = np.loadtxt("./data/simulation2_x_s.txt")
x = np.concatenate([x1, x2])
# with open('Utils/conf2.json', 'r') as f:
#     conf = json.load(f)
# ud = np.transpose(uts.fsl(100, np.array([50, 50]), 2)) / 100
# ud1 = ud[0:50, :]
# ud2 = ud[50:100, :]
# dds1 = Dds(conf, data=x1, label=y1)
# dds1.pca()
# x1_s, y1_s = dds1.sampling(ud1)
#
# dds2 = Dds(conf, data=x2, label=y2)
# dds2.pca()
# x2_s, y2_s = dds2.sampling(ud2)
#
#
with open('Utils/conf3.json', 'r') as f:
    conf = json.load(f)
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
# x_s = np.concatenate([x1_s, x2_s])
#
# dds3 = Dds(conf, data=x, label=y)
# dds3.pca()
#
# x3_s, y3_s = dds3.sampling(ud)
#
with open('Utils/conf3.json', 'r') as f:
    conf = json.load(f)
glp = uts.glp(100, 2)
dds4 = Dds(conf, data=x, label=y)
dds4.pca()
x4_s, y4_s = dds4.sampling(glp)

# g1 = uts.gefd(x, x_s)
# g2 = uts.gefd(x, x3_s)
# g11 = uts.fast_gefd(x, x_s)
# g22 = uts.fast_gefd(x, x3_s)
# np.savetxt("./data/simulation2_x1_s.txt", x1_s)
# np.savetxt("./data/simulation2_x2_s.txt", x2_s)
# np.savetxt("./data/simulation2_x3_s.txt", x3_s)
# np.savetxt("./data/simulation2_x_s.txt", x_s)

# plt.Figure()
# plt.scatter(ud1[:, 0], ud1[:, 1])
# plt.scatter(ud2[:, 0], ud2[:, 1], c = "r")
# plt.scatter(glp[:, 0], glp[:, 1], c = "g")
# plt.show(block=True)


fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[0].scatter(x1[:, 0], x1[:, 1], alpha=0.3, s=8)
ax[0].scatter(x1_s[:, 0], x1_s[:, 1], c="r", s=8)
ax[1].scatter(x2[:, 0], x2[:, 1], alpha=0.3, s=8)
ax[1].scatter(x2_s[:, 0], x2_s[:, 1], c="r", s=8)
plt.show(block =True)

fig2, ax2 = plt.subplots(1, 2, figsize = (10, 5))
ax2[0].scatter(x[:, 0], x[:, 1], alpha=0.3, s=8)
ax2[0].scatter(x_s[:, 0], x_s[:, 1], c="r", s=8)
ax2[1].scatter(x[:, 0], x[:, 1], alpha=0.3, s=8)
ax2[1].scatter(x3_s[:, 0], x3_s[:, 1], c="orange", s=8)
plt.show(block =True)

fig, ax = plt.subplots(2, 2, figsize = (12, 10))
ax[0, 0].scatter(x1[:, 0], x1[:, 1], alpha=0.3, s=8)
ax[0, 0].scatter(x1_s[:, 0], x1_s[:, 1], c="r", s=8)
ax[0, 1].scatter(x2[:, 0], x2[:, 1], alpha=0.3, s=8)
ax[0, 1].scatter(x2_s[:, 0], x2_s[:, 1], c="r", s=8)
ax[1, 0].scatter(x[:, 0], x[:, 1], alpha=0.3, s=8)
ax[1, 0].scatter(x_s[:, 0], x_s[:, 1], c="r", s=8)
ax[1, 1].scatter(x[:, 0], x[:, 1], alpha=0.3, s=8)
ax[1, 1].scatter(x3_s[:, 0], x3_s[:, 1], c="green", s=8)
plt.show(block =True)

fig1 = plt.Figure()
plt.scatter(x1[:, 0], x1[:, 1], alpha=0.4, s=20,edgecolors= 'none')
plt.scatter(x1_s[:, 0], x1_s[:, 1], c="r", s=8)
plt.title("$\mathcal{P}_1$ and $\mathcal{X}_1$" )
plt.show(block =True)

fig2 = plt.Figure()
plt.scatter(x2[:, 0], x2[:, 1], alpha=0.4, s=20, edgecolors= 'none')
plt.scatter(x2_s[:, 0], x2_s[:, 1], c="r", s=8)
plt.title("$\mathcal{P}_2$ and $\mathcal{X}_2$" )
plt.show(block =True)

fig3 = plt.Figure()
plt.scatter(x1[:, 0], x1[:, 1], alpha=0.4, s=20, edgecolors='none')
plt.scatter(x1_s[:, 0], x1_s[:, 1], c="r", s=8)
plt.scatter(x2[:, 0], x2[:, 1], alpha=0.4, s=20, edgecolors= 'none')
plt.scatter(x2_s[:, 0], x2_s[:, 1], c="r", s=8)
plt.title("$\mathcal{P}_1\coprod\mathcal{P}_1$ and $\mathcal{X}_1\coprod\mathcal{X}_2$" )
plt.show(block =True)

fig4 = plt.Figure()
plt.scatter(x[:, 0], x[:, 1], alpha=0.4, s=20, edgecolors='none')
plt.scatter(x3_s[:, 0], x3_s[:, 1], c="purple", s=8)
plt.title("$\mathcal{P}_{total}$ and $\mathcal{X}_1\coprod\mathcal{X}_2$" )
plt.show(block =True)


print('good luck!')