from numpy import *
import numpy as np
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':#左子树
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:#右子树
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)

    numSteps = 10.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min();#第一、2列的最小值
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps#计算步长
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                errArr = mat(ones((m, 1)))#全部设置为1
                errArr[predictedVals == labelMat] = 0#正确的设置为0，剩下的为错误的为1
                weightedError = D.T * errArr  # calc total error multiplied by D#错误和
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:#最后返回错误率最小情况下的决策树，错误率和类别估计值
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrain(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "error",error
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        print ("alpha",alpha)
        weakClassArr.append(bestStump)#store Stump Params in Array
        print ("classEst",classEst)
        #下面一句，如果某个样本被分错了，则yi * Gm(xi)为负，负负得正，结果使得整个式子变大（样本权值变大），否则变小
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon)) #Calc New D, element-wise
        D = D/D.sum()
        print ("D",D)
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print ("aggClassEst",aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        #print aggErrors
        errorRate = aggErrors.sum()/m
        print (errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr



def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)


datMat,classLabels = loadSimpData()
dataArr = mat(datMat)

classifierArr = adaBoostTrain(dataArr,classLabels,9)

print(adaClassify([0,0],classifierArr))

