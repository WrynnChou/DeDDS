import json
import numpy as np
from Utils.Utils import power_iter
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
import Utils.Utils as uts
import matplotlib.pyplot as plt

breast = load_breast_cancer()
breast_data = breast.data
breast_label = np.reshape(breast.target,(569,1))
final_breast_data = np.concatenate([breast_data,breast_label],axis=1)
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features_labels = np.append(features,'label')
breast_dataset.columns = features_labels


x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)
#PCA in Sklearn
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

#My pm PCA
x = torch.tensor(x)
A = x.transpose(0,1) @ x
vec, val = power_iter(A, 2, t_steps=10000)
prin = torch.as_tensor(x,dtype=torch.float32) @ vec
prin_1 = prin[:,0]
iecdf, funcs = uts.IECDF_nD(prin)
# x_p, y_p = np.meshgrid(np.arange(0,1,0.1),np.arange(0,1,0.1))
# points = torch.stack([torch.as_tensor(x_p), torch.as_tensor(y_p)], dim=-1).reshape(-1, 2)
# pp = iecdf(points)

UD = uts.glp(50, 2)
dim = UD.shape[1]
block = 5
data = prin

UD_inv = iecdf(UD)
idx_UD = uts.box_index(funcs, UD_inv, block)
neighbor_data, I = uts.data_index(idx_UD, data, block)
sub_prin, sub_idx = uts.nus_nnbrs(prin, iecdf, funcs, UD, 5)
sampling = x[sub_idx, :]
sampling_label = breast_label[sub_idx]

full_model = LogisticRegression()
sub_model = LogisticRegression()
full_model.fit(x, np.ravel(breast_label))
sub_model.fit(sampling, np.ravel(sampling_label))
yp_f = full_model.predict(x)
yp_s = sub_model.predict(x)
y = breast_label.reshape(569)
uts.eval(y,yp_s)
uts.eval(y,yp_f)

import  matplotlib.pyplot as plt
f = plt.Figure()
plt.scatter(x[:,0], x[:,1])
plt.show(block = True)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.figure(figsize=(10,10))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel('Principal Component - 1',fontsize=20)
# plt.ylabel('Principal Component - 2',fontsize=20)
# plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
# targets = ['Benign', 'Malignant']
# colors = ['r', 'g']
# for target, color in zip(targets,colors):
#     indicesToKeep = breast_dataset['label'] == target
#     plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
#                , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
#
# plt.legend(targets,prop={'size': 15})
# plt.show(block=True)
#
# a = 1
# print('ss')
# import  matplotlib.pyplot as plt
# plt.Figure()
# plt.scatter(prin[:,0],prin[:,1])
# plt.show(block = True)