
# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np


train = pd.read_csv('/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/train.csv',header=0)
test = pd.read_csv('/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/test.csv',header=0)
label = train.values[:, 0].astype(int)
train_data = train.values[:, 1:].astype(int)

test_data = test.values[:, :].astype(int)

pca = PCA(n_components=0.83, whiten=True).fit(train_data)

train_pca = pca.transform(train_data)
test_pca = pca.transform(test_data)

svc = SVC()
svc.fit(train_pca, label)


ans = svc.predict(test_pca)

a = []
for i in range(len(ans)):
    a.append(i+1)

np.savetxt('PCA_0.83_SVC.csv', np.c_[a, ans],
    delimiter=',', header='ImageId,Label', comments='', fmt='%d')