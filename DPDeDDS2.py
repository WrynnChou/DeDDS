
from algorithm.DDS import Dds
import numpy as np
import json

x = np.loadtxt("data/x_simulation.txt")
y = np.loadtxt('data/y_simulation.txt')
x_test = np.loadtxt("data/x_simulation_test.txt")
y_test = np.loadtxt("data/y_simulation_test.txt")

sud = np.loadtxt('temp/temp_glp.txt')
with open('Utils/conf4.json', 'r') as f:
    conf = json.load(f)

dds = Dds(conf, x, y)
dds.pca()
x_s, y_s = dds.sampling(sud)
dds.fit()
acc = dds.eval(x_test, y_test)
with open('temp/total.txt', 'a') as f:
    f.write("%s,\n" % (acc))

