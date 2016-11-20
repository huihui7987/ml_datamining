import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    '''

    :param inX: the input vector to classify待分类
    :param dataSet:our full matrix of training examples训练样本
    :param labels:a vector of labels
    :param k:he number of nearest neighbors to use in the voting.
    :return:
    '''
    dataSetSize = dataSet.shape[0]#行数
    #np.tile()用于扩充数组元素,下面将inX扩充为与训练样本一样的行数x1列的一个矩阵
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqdiffMat = diffMat ** 2
    sqDistances = sqdiffMat.sum(axis=1)#坐标轴x,横向相加（x^2+y^2）
    distance = sqDistances ** 0.5
    sortedDistIndicies = distance.argsort()#数组值从小到大的索引值，并非移动元素大小排序
    s= sortedDistIndicies[0]
    ss = sortedDistIndicies[1]
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]#labels[0]='A'
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

#group,labels = createDataSet()
#res = classify0([0.5,0],group,labels,3)
#print(res)

def file2matrix(filename):
    fr = open(filename)
    arrayOnlines = fr.readlines()
    numberOfLines = len(arrayOnlines)
    returnMat = np.zeros((numberOfLines,3))#0矩阵，3列
    classLabelVector = []
    index = 0
    for line in arrayOnlines:
        line = line.strip()
        listFromLine = line.split('\t')
        #每行
        returnMat[index,:] = listFromLine[0:3]#3列
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


filename = '/Users/huihui/PycharmProjects/machinelearninginaction/Ch02/datingTestSet2.txt'
retutnMat,classLabelVector = file2matrix(filename)
#print(retutnMat,classLabelVector)

#画散点图
#import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(retutnMat[:,2],retutnMat[:,0],
           15.0*np.array(classLabelVector),
           15.0 * np.array(classLabelVector))
#plt.show()

def autoNorm(dataSet):
    #uu= np.shape(dataSet)
    minVals = dataSet.min(0)#最小值
    maxVals = dataSet.max(0)#最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))#行列数
    m = dataSet.shape[0]#1000
    #hh = np.tile(minVals,(m,1))构造相同行数的矩阵便于计算

    normDataSet = (dataSet-np.tile(minVals,(m,1)))/(np.tile(ranges,(m,1)))

    return normDataSet,ranges,minVals

#print(autoNorm(retutnMat))

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]#1000,行数

    numTestVecs = int(m*hoRatio)#十分之一的测试，90的学习
    errorCount = 0.0
    for i in range(numTestVecs):
        #s = normMat[i, :]
        #ss = normMat[numTestVecs:m, :]#100以后的是训练集
        #sss =  datingLabels[numTestVecs:m]
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)
res = datingClassTest()
print(res)


def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']

    ffMiles = float(input("frequent flier miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))

    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,10)
    print ("You will probably like this person: ",resultList[classifierResult - 1])


#res = classifyPerson()
print(res)















