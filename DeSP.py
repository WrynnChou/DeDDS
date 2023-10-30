import json
import time
import numpy as np
from mpi4py import MPI
from algorithm.SP import Sp
import sklearn
import Utils.Utils as uts

num = "3"
sim = "a"

t1 = time.perf_counter()
# get MPI info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

<<<<<<< HEAD
x = np.loadtxt("data/x_simulation.txt")
y = np.loadtxt('data/y_simulation.txt')
=======
x = np.loadtxt("data/x_simulation" + num + ".txt")
y = np.loadtxt('data/label_simulation' + num + '.txt')
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

with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

iteration = 1000
for j in range(iteration):
    sp = Sp(conf, data=x_l, label=y_l)
<<<<<<< HEAD
    sp.sampling(k = 150)
    x_s, y_s =sp.nnbrs()
=======
    sp.sampling(k = 120)
    x_s, y_s = sp.nnbrs()
>>>>>>> 461b612 (README)
    combine_x = np.array(comm.gather(x_s, root=0))
    combine_y = np.array(comm.gather(y_s, root=0))

    # print("This is note %s." % (rank))
    sp.fit()
    acc = sp.eval(x_tl, y_tl)
    for i in range(size):
        if i == rank:
<<<<<<< HEAD
            with open('temp/desp1.txt', 'a') as f:
=======
            with open('temp/desp' + num + sim + '.txt', 'a') as f:
>>>>>>> 461b612 (README)
                f.write("%s," % acc)
        comm.barrier()
    # if rank == 0:
    #     sp.plot_2D()
    comm.barrier()
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
        acc2 = uts.eval(model.predict(x_test), y_test, False)
        # with open('logs/simulation.txt', 'a') as f:
        #     f.write("Total SP, acc: %.2f%%.\n" % (acc2))
<<<<<<< HEAD
        with open('temp/desp1.txt', 'a') as f:
=======
        with open('temp/desp' + num + sim + '.txt', 'a') as f:
>>>>>>> 461b612 (README)
            f.write("%s\n" % acc2)
    comm.barrier()

t2 = time.perf_counter()
if rank == 0:
    print("Time cost %ss" % (t2 - t1))
    print("Have a nice day!")
