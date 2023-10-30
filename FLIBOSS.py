import pandas as pd
from torch.utils.data import DataLoader
from model import *
import json
import numpy as np
import sklearn
from mpi4py import MPI
from algorithm.IBOSS import Iboss
import Utils.Utils as uts

# get MPI info
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

with open('Utils/conf5.json', 'r') as f:
    conf = json.load(f)

path = 'data/Real_dataset/' + str(rank + 2) + '.xlsx'
xx = pd.read_excel(path)
df = xx.iloc[:, 3:10].fillna(0)
x = df.values[:, 1:7]
y = df.values[:, 0]
local_epoch = conf["local_epoch"]
global_epoch = conf["global_epoch"]
batch = conf["batch"]
lr = conf["learning_rate"]
lamb = torch.tensor(1 / size)

iboss = Iboss(conf=conf, data=x, label=y)
x_s, y_s = iboss.sampling()

save_path = 'logs/iboss_air_x' + str(rank) + '.txt'
save_path2 = 'logs/iboss_air_y' + str(rank) + '.txt'

np.savetxt(save_path, x_s)
np.savetxt(save_path2, y_s)

combine_x = np.array(comm.gather(x_s, root=0))
combine_y = np.array(comm.gather(y_s, root=0))

torch.manual_seed(777)
loss_function = nn.MSELoss(reduction='mean')
global_model = Ann()
global_opt = torch.optim.Adam(global_model.parameters(), lr=lr)
local_model = Ann()
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
        print(loss_g)

if rank == 0:
    path2 = 'data/Real_dataset/26.xlsx'
    xxt = pd.read_excel(path2)
    dft = xxt.iloc[:, 3:10].fillna(0)
    x_test = torch.from_numpy(dft.values[:, 1:7]).float()
    y_test = torch.unsqueeze(torch.from_numpy(dft.values[:, 0]), 1).float()

    out_test = global_model.model(x_test)
    loss_t = loss_function(out_test, y_test)
    print(loss_t)


print("Have a nice day!")
