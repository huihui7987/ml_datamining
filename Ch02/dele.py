
import pandas as pd
import numpy as np
import csv as csv
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



trainDataSet = pd.read_csv('/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/train.csv',header=0)
testDataSet = pd.read_csv('/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/test.csv',header=0)

#knn = KNeighborsClassifier()

train = trainDataSet.values
test = testDataSet.values
train_data = train[:,1:]
train_label = train[:,0]
test_data = test[:,:]

print('start training ......')

pca = PCA(n_components=0.95, whiten=True).fit(train_data)

train_pca = pca.transform(train_data)
test_pca = pca.transform(test_data)

svc = SVC()
svc.fit(train_pca,train_label)

print('start predciting :')


out = svc.predict(test_pca)

print('start writing: ')

n,m = test.shape
ids = range(1,n+1)

predictions_file = open('outddd.csv','w')
open_file_object = csv.writer(predictions_file)
open_file_object.writerows(['ImageId'],['Lables'])
open_file_object.writerows(zip(ids,out))

predictions_file.close()

print('All is DONE')

