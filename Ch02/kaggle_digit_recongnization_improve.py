'''

最终提交的结果为96.8，时间大概半小时。考虑PCA降维
'''
import pandas as pd
import numpy as np
import csv as csv

from sklearn.neighbors import KNeighborsClassifier

trainDataSet = pd.read_csv('/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/train.csv',header=0)
testDataSet = pd.read_csv('/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/test.csv',header=0)

knn = KNeighborsClassifier()

train = trainDataSet.values
test = testDataSet.values

print('start training ......')

knn.fit(train[0:,1:],train[0:,0])

print('start predciting :')


out = knn.predict(test)

print('start writing: ')

n,m = test.shape
ids = range(1,n+1)

predictions_file = open('out.csv','w')
open_file_object = csv.writer(predictions_file)
open_file_object.writerows(['ImageId','Lables'])
open_file_object.writerows(zip(ids,out))

predictions_file.close()

print('All is DONE')

