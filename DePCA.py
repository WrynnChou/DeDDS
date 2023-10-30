import numpy as np
import scipy
import Utils.Utils as uts
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import Consensus
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from sklearn.linear_model import LogisticRegression


# get MPI info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

x = np.loadtxt("data/x.txt")
y = np.loadtxt("data/y.txt").reshape(569)


batch = np.floor(np.shape(x)[0] / size)
start = np.int32(rank * batch )
end = np.int32((rank + 1) * batch )
x_l = x[start : end ,:]


Adj = binomial_random_graph(size, p=0.3, seed=1)
W = metropolis_hastings(Adj)
# reset local seed
np.random.seed()

a = np.transpose(x_l) @ x_l
w0 = np.ones((30,2))
iteration = 10
# # create local agent
agent = Agent(in_neighbors=np.nonzero(Adj[rank, :])[0].tolist(),
              out_neighbors=np.nonzero(Adj[:, rank])[0].tolist(),
              in_weights=W[rank, :].tolist())

for i in range(iteration):
    w0 = a @ w0
    algorithm = Consensus(agent=agent,
                          initial_condition=w0,
                          enable_log=True)
    sequence = algorithm.run(iterations=5)
    w0 = algorithm.get_result()
    q, r = scipy.linalg.qr(w0, mode = "economic")
    w0 = q
pca_l = x_l @ w0

comm.barrier()

if rank == 0:
    pca_dec = x @ w0
    data = pca_dec
    iecdf, funcs = uts.IECDF_nD(pca_dec)
    UD = uts.glp(50, 2)
    sub_prin, sub_idx = uts.nus_nnbrs(pca_dec, iecdf, funcs, UD, 5)

    subsampling = x[sub_idx,:]
    subsampling_y = y[sub_idx]
    sub_model = LogisticRegression()
    sub_model.fit(subsampling,subsampling_y)
    y_p = sub_model.predict(x)
    uts.eval(np.ravel(y), y_p)

    import  matplotlib.pyplot as plt
    plt.Figure()
    plt.scatter(x[:, 0], x[:, 1])
    plt.scatter(subsampling[:, 0], subsampling[:, 1], c="r")
    plt.show(block=True)


 