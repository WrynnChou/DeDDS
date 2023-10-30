import json
import pandas as pd
from mpi4py import MPI
from algorithm.DeDDS import Dedds
from algorithm.DP import *
from torch.utils.data import DataLoader
from model import *
from Utils.Utils import *
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings



# get MPI info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

with open('Utils/conf5.json', 'r') as f:
    conf = json.load(f)

x = np.loadtxt('data/air/' + str(rank + 1987) + '.txt')
y = np.loadtxt('data/air/' + str(rank + 1987) + '_y.txt')

local_epoch = conf["local_epoch"]
global_epoch = conf["global_epoch"]
batch = conf["batch"]
lr = conf["learning_rate"]
lamb = torch.tensor(1 / size)

Adj = binomial_random_graph(size, p=0.3, seed=1)
W = metropolis_hastings(Adj)

ud_path = "./temp/glp/" + str(conf['subsampling_number']) + "/temp_glp_" + str(0) + ".txt"
sliced_ud = np.loadtxt(ud_path)

# reset local seed
np.random.seed()

dedds = Dedds(conf=conf, data=x, label=y, comm=comm)

agent = dedds.creat_agent(Adj, W)
pca_comp = dedds.try_pca()
pca_l, w0 = dedds.deepca(300)

# DP histogram
max_list = []
min_list = []

for i in range(pca_comp):
    max_ = np.max(pca_l[:, i])
    min_ = np.min(pca_l[:, i])
    max_all = comm.reduce(max_, MPI.MAX)
    min_all = comm.reduce(min_, MPI.MIN)
    max_all = comm.bcast(max_all)
    min_all = comm.bcast(min_all)
    max_list.append(max_all)
    min_list.append(min_all)
    comm.barrier()

hist_list, bins_list = [], []
for i in range(pca_comp):
    hist, bins = hist_est(pca_l[:, i], np.array([min_list[i], max_list[i]]), plot=False, num=20, density=False)
    hist_temp = comm.reduce(hist, MPI.SUM)
    hist_temp = comm.bcast(hist_temp)
    bins_list.append(bins)
    hist_list.append(hist_temp)
    comm.barrier()

hist_list = hist_to_ecdf(hist_list)

# use liner function of histogram to estimate iecdf
iecdf, funcs = liner_inv_nd(hist_list, bins_list)

# DP inverse function

save_path = 'logs/dedds_air_x' + str(rank) + '.txt'
save_path2 = 'logs/dedds_air_y' + str(rank) + '.txt'

x_s, y_s = dedds.sampling_dpconsesus(sliced_ud, iecdf, funcs, 3)
np.savetxt(save_path, x_s)
np.savetxt(save_path2, y_s)

combine_x = np.array(comm.gather(x_s, root=0))
combine_y = np.array(comm.gather(y_s, root=0))

torch.manual_seed(777)
loss_function = nn.MSELoss(reduction='mean')
global_model = Ann2()
global_opt = torch.optim.Adam(global_model.parameters(), lr=lr)
local_model = Ann2()
local_opt = torch.optim.Adam(local_model.parameters(), lr=lr)

x_st = torch.from_numpy(x_s).float()
y_st = torch.unsqueeze(torch.from_numpy(y_s), 1).float()
x_t = torch.from_numpy(x).float()
y_t = torch.unsqueeze(torch.from_numpy(y), 1).float()
data_ = torch.utils.data.TensorDataset(x_st, y_st)
dataloader = DataLoader(data_, batch_size=batch)

for i in range(global_epoch):

    # download local model
    for name, para in global_model.state_dict().items():
        local_model.state_dict()[name].copy_(para.clone())

    # local update
    for j in range(local_epoch):
        for para in dataloader:
            x, y = para
            output = local_model.model(x)
            loss = loss_function(output, y)

            local_opt.zero_grad()
            loss.backward()
            local_opt.step()

    # aggregate
    diff = dict()
    for name, para in local_model.state_dict().items():

        diff_l = (para - global_model.state_dict()[name])
        diff_ = comm.reduce(diff_l, MPI.SUM)
        diff_ = comm.bcast(diff_)
        diff[name] = torch.clone(diff_) * lamb

    # global update
    for name, para in global_model.state_dict().items():
        para.add_(diff[name])

    if rank == 0:
        out = global_model.model(x_t)
        loss_g = loss_function(y_t, out)

    comm.barrier()

if rank == 0:

    x_test = torch.from_numpy(np.loadtxt('data/air/2006.txt')).float()
    y_test = torch.unsqueeze(torch.from_numpy(np.loadtxt('data/air/2006_y.txt')), 1).float()

    out_test = global_model.model(x_test)
    loss_t = loss_function(out_test, y_test)
    print(loss_t)


print("Have a nice day!")
