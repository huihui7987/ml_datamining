from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)#5
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

dataSet = createDataSet()[0]
labels = createDataSet()[1]#
#res = calcShannonEnt(dataSet)
#print(res)
'''
self,same as the above
'''
'''
def calcShannonent(dataSet):
    nument = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount:
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1

    shannonent = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / nument
        shannonent -= prob*log(prob,2)
    return shannonent


res = calcShannonent(dataSet)
print(res)

'''
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:#条件，
            reducedFeatVec = featVec[:axis]#axis = 0时，为[],就是list切片

            reducedFeatVec.extend(featVec[axis+1:])#以上两行代码就是，某个属性用过之后就从属性集中删除data[:x]+data[x+1:] = data-data[x],x为所选属性
            retDataSet.append(reducedFeatVec)
    return retDataSet

#rre = splitDataSet(dataSet,1,0)

#print(rre)

def chooseBestFeatureToSplit(dataSet):#ID3求信息增益Ginfo
    numFeature = len(dataSet[0]) - 1#最后一列为分类
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeature):#遍历每个属性#年龄，收入，学历，人口
        featList = [ex[i] for ex in dataSet]
        uniqueVals = set(featList)#青年，中年，老年
        newEntropy = 0.0

        for value in uniqueVals:#遍历每个子属性，为求newEntropy
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infogain = baseEntropy - newEntropy
        if infogain > bestInfoGain:
            bestInfoGain = infogain
            bestFeature = i
    return bestFeature


import operator
def majoruityCnt(classList):#返回出现次数最多的类
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]



def creatTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#类别完全相同
        return classList[0]
    if len(dataSet[0]) == 1:
        return majoruityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet,bestFeat,value),subLabels)

    return myTree




res = creatTree(dataSet,labels)
print(res)














