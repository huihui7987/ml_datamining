#-*- coding:utf-8 -*-

import numpy as np

filename = '/Users/huihui/PycharmProjects/ml_datamining/Ch05/testSet.txt'

def loadDataSet(filename):
    '''
    从文件中读取数据
    :param filename:
    :return:
    '''
    dataMat = []
    labelMat = []
    with open(filename,errors='ignore') as f:
        lines = f.readlines()
    for line in lines:
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return  dataMat,labelMat

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscend(dataMatIn,classLabels):

    '''

    :param dataMatIn:
    :param classLabels:
    :return:
    '''

    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)#m,n = dataMatrix.shape
    alpha = 0.001
    maxCycle = 500
    weights = np.ones((n,1))

    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h )
        weights = weights + alpha * dataMatrix.transpose()*error

    return weights



# def stocGradAscent0(dataMatrix,classLabels):
#     m,n = dataMatrix.shape
#     alpha = 0.01
#     weights = np.ones(n)
#     for i in range(m):
#         h = sigmoid(sum(dataMatrix[i]*weights))
#         error = classLabels[i]-h
#         weights = weights + alpha * error * dataMatrix[i]
#     return weights



def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

dataMat,labelMat = loadDataSet(filename)
res = stocGradAscent0(np.array(dataMat),labelMat)

print(res)


import matplotlib.pyplot as plt

def plotestFit(wei):
    '''

    :param wei:
    :return:
    '''
    weights = wei#.getA()#return self as ndarray object
    dataMat,labelMat = loadDataSet(filename)
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x1 = np.arange(-3.0,3.0,0.1)
    '''
    set0 = w0x0 + w1x1 + w2x2 and solved for X2 in terms of X1 (remember, X0 was 0)
    也就是解x1与x2 之间的一次关系
    '''
    x2 = (-weights[0]-weights[1]*x1) / weights[2]#
    ax.plot(x1,x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


plotestFit(res)


#fsd





















