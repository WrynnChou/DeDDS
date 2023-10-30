import numpy as np
import Utils.Utils as uts
from mpi4py import MPI
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from algorithm.DeOptPoisson import Deoptpoisson
import json
import sklearn

num = "2"
sim = "a"

# get MPI info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

x = np.loadtxt(("data/x_simulation" + num + ".txt"))
y = np.loadtxt("data/label_simulation" + num + ".txt")

batch = np.floor(np.shape(x)[0] / size)
start = np.int32(rank * batch)
end = np.int32((rank + 1) * batch)
x_l = x[start:end, :]
y_l = y[start:end]

x_test = np.loadtxt("data/x_simulation_test" + num + ".txt")
y_test = np.loadtxt("data/label_simulation_test" + num + ".txt")
batch_t = np.floor(np.shape(x_test)[0] / size)
start_t = np.int32(rank * batch_t)
end_t = np.int32((rank + 1) * batch_t)
x_tl = x_test[start_t:end_t, :]
y_tl = y_test[start_t:end_t]

# reset local seed
np.random.seed()

iteration = 100
for j in range(iteration):
    optpoisson = Deoptpoisson(conf, x_l, y_l, comm)
    pre_samp, pre_samp_y = optpoisson.pre_sampling()
    beta_0, psi = optpoisson.qse()
    x_s, y_s = optpoisson.sampling()

    combine_x = np.array(comm.gather(x_s, root=0))
    combine_y = np.array(comm.gather(y_s, root=0))

    comm.barrier()
    optpoisson.fit()
    acc = optpoisson.eval(x_tl, y_tl)
    for i in range(size):
        if i == rank:
            with open(('temp/deopt' + num + sim + '.txt'), 'a') as f:
                f.write("%s," % (acc))
        comm.barrier()

    if rank == 0:
        combine_x_ = np.concatenate(combine_x)
        combine_y_ = np.concatenate(combine_y)
        model = sklearn.linear_model.LogisticRegression()
        model.fit(combine_x_, combine_y_)
        acc2 = uts.eval(model.predict(x_test), y_test, False)
        with open('temp/deopt' + num + sim + '.txt', 'a') as f:
            f.write("%s\n" % acc2)

print('Have a nice day!')