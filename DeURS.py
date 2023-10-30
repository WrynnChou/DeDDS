import json
import numpy as np
from mpi4py import MPI
from algorithm.URS import Urs
import sklearn
import Utils.Utils as uts

num = "3"
sim = "a"

# get MPI info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

<<<<<<< HEAD
x = np.loadtxt("data/x_simulation.txt")
y = np.loadtxt('data/y_simulation.txt')
=======
x = np.loadtxt(("data/x_simulation" + num + ".txt"))
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

with open('Utils/conf.json', 'r') as f:
    conf = json.load(f)

iteration = 100
for j in range(iteration):
    urs = Urs(conf=conf, data=x_l, label=y_l)
    x_s, y_s = urs.sampling()
    combine_x = np.array(comm.gather(x_s, root=0))
    combine_y = np.array(comm.gather(y_s, root=0))

    # print("This is note %s." % (rank))
    urs.fit()
    acc = urs.eval(eval_data=x_tl, eval_label=y_tl)
    for i in range(size):
        if i == rank:
            # with open('logs/simulation.txt', 'a') as f:
            #     f.write("Note %s, acc: %.2f%%." % (rank, acc))
<<<<<<< HEAD
            with open('temp/urs1.txt', 'a') as f:
=======
            with open(('temp/urs' + num + sim + '.txt'), 'a') as f:
>>>>>>> 461b612 (README)
                f.write("%s," % (acc))
        comm.barrier()

    # if rank == 0:
    #     urs.plot_2D()
    comm.barrier()
    if rank == 0:
        combine_x_ = np.concatenate(combine_x)
        combine_y_ = np.concatenate(combine_y)
        model = sklearn.linear_model.LogisticRegression()
        model.fit(combine_x_, combine_y_)
        # print("This is total model:")
        acc2 = uts.eval(model.predict(x_test), y_test, pit=False)
        # with open('logs/simulation.txt', 'a') as f:
        #     f.write("Total URS, acc: %.2f%%.\n" % (acc2))
<<<<<<< HEAD
        with open('temp/urs1.txt', 'a') as f:
=======
        with open('temp/urs' + num + sim + '.txt', 'a') as f:
>>>>>>> 461b612 (README)
            f.write("%s\n" % acc2)

print("Have a nice day!")