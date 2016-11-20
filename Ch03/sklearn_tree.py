from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

data = load_iris()
feature = data['data']
feature_names = data['feature_names']
target = data['target']

'''
def plot_iris_projection(x_index, y_index):
    for t,marker,c in zip(range(3),'>ox', 'rgb'):
        plt.scatter(feature[target==t,x_index],
                    feature[target==t,y_index],
                    marker=marker,c=c)
        plt.xlabel(feature_names[x_index])
        plt.ylabel(feature_names[y_index])

pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
for i,(x_index,y_index) in enumerate(pairs):
    plt.subplot(2,3,i+1)
    plot_iris_projection(x_index, y_index)
plt.show()
'''
dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
labels = ['no surfacing','flippers']




datalist = [ex[0:2] for ex in dataSet]
datavec = [ex[-1] for ex in dataSet ]
filename = '/Users/huihui/PycharmProjects/machinelearninginaction/Ch03/lenses.txt'


def createDataSetFromFile(filename):
    with open (filename) as fr:
        lenses = [line.strip().split('\t') for line in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses,lensesLabels



lenses,lenseLabels = createDataSetFromFile(filename)
classList = [ex[-1] for ex in lenses]
dataList = [ex[0:4] for ex in lenses]
print(dataList)
print(classList)
#[['young', 'myope', 'no', 'reduced'], ['young', 'myope', 'no', 'normal'], ['young', 'myope', 'yes', 'reduced'], ['young', 'myope', 'yes', 'normal'], ['young', 'hyper', 'no', 'reduced'], ['young', 'hyper', 'no', 'normal'], ['young', 'hyper', 'yes', 'reduced'], ['young', 'hyper', 'yes', 'normal'], ['pre', 'myope', 'no', 'reduced'], ['pre', 'myope', 'no', 'normal'], ['pre', 'myope', 'yes', 'reduced'], ['pre', 'myope', 'yes', 'normal'], ['pre', 'hyper', 'no', 'reduced'], ['pre', 'hyper', 'no', 'normal'], ['pre', 'hyper', 'yes', 'reduced'], ['pre', 'hyper', 'yes', 'normal'], ['presbyopic', 'myope', 'no', 'reduced'], ['presbyopic', 'myope', 'no', 'normal'], ['presbyopic', 'myope', 'yes', 'reduced'], ['presbyopic', 'myope', 'yes', 'normal'], ['presbyopic', 'hyper', 'no', 'reduced'], ['presbyopic', 'hyper', 'no', 'normal'], ['presbyopic', 'hyper', 'yes', 'reduced'], ['presbyopic', 'hyper', 'yes', 'normal']]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(dataList, classList)
print(clf)
print(clf.predict([['young', 'myope', 'no', 'reduced']]))

    #lenseTree = createTree(lenses,lenseLabels)

    #createPlotRES = createPlot(lenseTree)
    #return createPlotRES







