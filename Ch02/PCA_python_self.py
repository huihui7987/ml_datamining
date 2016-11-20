import numpy as np

#dataSet = []

def zeroMean(dataSet):#dataSet 一行代表一个样本，一列代表一个属性
    meanVal = np.mean(dataSet,axis = 0)#按列求均值,axis=0
    newDataSet = dataSet - meanVal
    return newDataSet,meanVal
#求协方差矩阵


def pca(dataMat,k):
    newDataSet,meanVal = zeroMean(dataMat)
    covMat = np.cov(newDataSet,rowvar=0)#rowvar=0,说明传入的数据一行代表一个样本。若不等于0，说明一列代表一个样本
    #ss = covMat
    #sss = np.mat(covMat)
    #求特征值与特征向量
    eigVals,eigVect = np.linalg.eig(np.mat(covMat))#直接求特征值与特征向量，eigVals行向量，特征向量已经转置为列向量，每一列代表一个特征向量

    #保留特征值的大小排序的前k个
    #k = 3
    eigValIndice = np.argsort(eigVals)#从小到大排序,返回的是特征值的下脚标，而不是排名
    k_eigValIndice = eigValIndice[-1:-(k+1):-1]#list[a:b:c]表示a-b,步长为c，负号表示大到小,最大的K个特征值的下标

    k_eigVect = eigVect[:,k_eigValIndice]#对应的特征向量

    lowDDataMat = newDataSet * k_eigVect#得到降维后的矩阵
    #reconMat = (lowDDataMat * k_eigVect.T)+meanVal#重构数据，后续或许用不到

    return lowDDataMat#,reconMat



def percentage2n(eigVals,percentage):
    sortArray = np.sort(eigVals)#升序
    sortArray = sortArray[-1::-1]#反转
    arraySum = sum(sortArray)#特征值求和？
    #temSum = 0
    num = 0
    for i in sortArray:
        #temSum += 1
        num+=1
        if num >= arraySum * percentage:
            return num


#array = [1.76815970e+00,2.48506964e-01,4.39791749e-18]
#res = percentage2n(array,0.95)
#print(res)


def pcaPercentage(dataMat,percentage=0.99):#默认0.99
    newDataSet,meanVal = zeroMean(dataMat)
    covMat = np.cov(newDataSet,rowvar=0)#rowvar=0,说明传入的数据一行代表一个样本。若不等于0，说明一列代表一个样本
    #ss = covMat
    #sss = np.mat(covMat)
    #求特征值与特征向量
    eigVals,eigVect = np.linalg.eig(np.mat(covMat))#直接求特征值与特征向量，eigVals行向量，特征向量已经转置为列向量，每一列代表一个特征向量

    #保留特征值的大小排序的前k个
    #k = 3
    eigValIndice = np.argsort(eigVals)#从小到大排序,返回的是特征值的下脚标，而不是排名
    k = percentage2n(eigVals,percentage)
    k_eigValIndice = eigValIndice[-1:-(k+1):-1]#list[a:b:c]表示a-b,步长为c，负号表示大到小,最大的K个特征值的下标

    k_eigVect = eigVect[:,k_eigValIndice]#对应的特征向量

    lowDDataMat = newDataSet * k_eigVect#得到降维后的矩阵
    #reconMat = (lowDDataMat * k_eigVect.T)+meanVal#重构数据，后续或许用不到

    return lowDDataMat#,reconMat

#res = pcaPercentage([[1.1,3,1.1],[2.1,3.4,0.5],[0.9,1.1,0.7]],0.94)

#print(res)

import numpy as np
import pandas as pd
import operator
import csv
trainfilename = '/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/train.csv'
testfilename = '/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/test.csv'
def load_trainDataSet(filename):
    l = []
    with open(filename) as file:
        lines =  csv.reader(file)
        for line in lines:
            l.append(line)

    l = l[1:-1]
    l = np.array(l)
    labels = l[:,0]
    trainDataSet = l[:,1:]
    trainDataSet = trainDataSet.astype(int)#将string 转化为 整数

    '''
    将非0得像素值转为1，只留下二值，便于计算
    '''
    trainDataSet = nomalizing(trainDataSet)
    #ss = trainDataSet

    return trainDataSet,labels



def nomalizing(array):
    m,n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i][j] != 0:
                array[i][j] = 1

    return array

#def saveResult(result):
#    with open('uuuui.csv','w') as myFile:
#        myWriter = csv.writer(myFile)
#        for i in result:
#            tmp = []
#            tmp.append(i)
#            myWriter.writerow(tmp)



rr = [[-0.34260228,0.56111131],
 [-1.12489617,-0.39178757],
 [ 1.46749845,-0.16932374]]



def saveResult(result):
    m, n = np.shape(result)
    with open('pcauuuu.csv', 'w') as myFile:
        myWriter = csv.writer(myFile)
        for i in range(m):
            #res = result[i]
            myWriter.writerow(result[i])


trainDataSet,labels = load_trainDataSet(trainfilename)
res = pcaPercentage(trainDataSet,0.95)
saveResult(res)
print(res)