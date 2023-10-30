import numpy as np
import Utils.Utils as uts
from Utils.Utils import *
from mpi4py import MPI
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from algorithm.DeDDS import Dedds
from algorithm.DP import *
import json
import sklearn
import time

num = "3"
sim = "a"

t1 = time.perf_counter()
# get MPI info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

<<<<<<< HEAD
x = np.loadtxt("data/x_simulation.txt")
y = np.loadtxt("data/y_simulation.txt")
=======
x = np.loadtxt("data/x_simulation" + num + ".txt")
y = np.loadtxt("data/label_simulation" + num + ".txt")
>>>>>>> 461b612 (README)

batch = np.floor(np.shape(x)[0] / size)
start = np.int32(rank * batch)
end = np.int32((rank + 1) * batch)
x_l = x[start:end, :]
y_l = y[start:end]

<<<<<<< HEAD
x_test = np.loadtxt("data/x_simulation_test.txt")
y_test = np.loadtxt("data/y_simulation_test.txt")
=======
x_test = np.loadtxt("data/x_simulation_test" + num + ".txt")
y_test = np.loadtxt("data/label_simulation_test" + num + ".txt")
>>>>>>> 461b612 (README)
batch_t = np.floor(np.shape(x_test)[0] / size)
start_t = np.int32(rank * batch_t)
end_t = np.int32((rank + 1) * batch_t)
x_tl = x_test[start_t:end_t, :]
y_tl = y_test[start_t:end_t]

# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(size, p=0.3, seed=1)
W = metropolis_hastings(Adj)

<<<<<<< HEAD
sliced_ud = np.loadtxt('temp/temp_glp.txt')
=======
>>>>>>> 461b612 (README)
# if rank == 0:
#     print(Adj)
#     print(W)
# reset local seed
np.random.seed()

<<<<<<< HEAD
iteration = 1
for j in range(iteration):
=======
iteration = 50
for j in range(iteration):

    ud_path = "./temp/glp/" + str(conf['subsampling_number']) + "/temp_glp_" + str(j) + ".txt"
    sliced_ud = np.loadtxt(ud_path)


>>>>>>> 461b612 (README)
    dedds = Dedds(conf, x_l, y_l, comm)
    agent = dedds.creat_agent(Adj, W)
    pca_comp = dedds.try_pca()
    pca_l, w0 = dedds.deepca(100)

    pca_all = np.array(comm.gather(pca_l[:, 0]))
    if rank == 0:
        pca_all = np.concatenate(pca_all)
        f = ecdf_1dimension(pca_all)

    # DP histogram
    max_list = []
    min_list = []

    for i in range(pca_comp):
        max_ = max(pca_l[:, i])
        min_ = min(pca_l[:, i])
        max_all = comm.reduce(max_, MPI.MAX)
        min_all = comm.reduce(min_, MPI.MIN)
        max_all = comm.bcast(max_all)
        min_all = comm.bcast(min_all)
        max_list.append(max_all)
        min_list.append(min_all)
        comm.barrier()

    hist_list, bins_list = [], []
    for i in range(pca_comp):
        hist, bins = hist_est(pca_l[:, i], np.array([min_list[i], max_list[i]]), epsilon=0.2 ,plot=False, num= 50, density=False)
        hist_temp = comm.reduce(hist, MPI.SUM)
        hist_temp = comm.bcast(hist_temp)
        bins_list.append(bins)
        hist_list.append(hist_temp)
        comm.barrier()

    hist_list = hist_to_ecdf(hist_list)
    # direct use histogram to estimate iecdf
    # iecdf, funcs = hist_iecdf_nd(hist_list, bins_list)

    # use liner function of histogram to estimate iecdf
    iecdf, funcs = liner_inv_nd(hist_list, bins_list)

    # cdf plot
    # cdf_list = []
    # for j in range(pca_comp):
    #     dens_dp = bins_list[j] / sum(bins_list[j])
    #     cdf_dp = np.zeros_like(dens_dp)
    #     a = 0
    #     for i in range(len(dens_dp)):
    #         cdf_dp[i] = a
    #         a += dens_dp[i]
    #     cdf_list.append(cdf_dp)
    # if rank == 0:
    #     xxx = np.linspace(min_list[0], max_list[0], 10000)
    #     yyy = np.zeros_like(xxx)
    #     for i in range(10000):
    #         yyy[i] = f(xxx[i])
    #     plt.figure()
    #     plt.plot(xxx, yyy)
    #     plt.plot(hist_list[0], cdf_list[0], c = "orange")
    #     plt.show(block=True)



    # sliced ud
    # if rank == 0:
    #     m_l = dedds.n * (np.ones(size).astype("int"))
    #     ud = uts.fsl(int(dedds.n * size), m_l, dedds.prin_comp)
    #     sud = uts.divi(ud, size)
    # else:
    #     sud = None
    # sliced_ud = comm.scatter(sud, root=0)

    # DP inverse function
    x_s, y_s = dedds.sampling_dpconsesus(sliced_ud, iecdf, funcs)

    # direct use local inverse function with DP
    # x_s, y_s = dedds.sampling_single(sliced_ud, 5)


    combine_x = np.array(comm.gather(x_s, root=0))
    combine_y = np.array(comm.gather(y_s, root=0))
    dedds.fit()
    # print("This is note %s." % (rank))
    acc = dedds.eval(x_tl, y_tl, False)

    for i in range(size):
        if i == rank:
<<<<<<< HEAD
            with open('temp/dpdedds4.txt', 'a') as f:
=======
            with open('temp/dpdedds' + num + sim + '.txt', 'a') as f:
>>>>>>> 461b612 (README)
                f.write("%s," % acc)
        comm.barrier()

    # if rank == 0:
    #     dedds.plot_2D()
    if rank == 0:
<<<<<<< HEAD
        x_test = np.loadtxt("data/x_simulation_test.txt")
        y_test = np.loadtxt("data/y_simulation_test.txt")
=======
        x_test = np.loadtxt("data/x_simulation_test" + num + ".txt")
        y_test = np.loadtxt("data/label_simulation_test" + num + ".txt")
>>>>>>> 461b612 (README)
        combine_x_ = np.concatenate(combine_x)
        combine_y_ = np.concatenate(combine_y)
        model = sklearn.linear_model.LogisticRegression()
        model.fit(combine_x_, combine_y_)
        # print("This is total model:")
        acc2 = uts.eval(model.predict(x_test), y_test)
<<<<<<< HEAD
        with open('temp/dpdedds4.txt', 'a') as f:
=======
        with open('temp/dpdedds' + num + sim + '.txt', 'a') as f:
>>>>>>> 461b612 (README)
            f.write("%s\n" % acc2)
    comm.barrier()

t2 = time.perf_counter()
if rank == 0:
    print("Time cost %ss" % (t2 - t1))
    print("Have a nice day!")