import os,random

def get_dataSet():
    data = []
    for root,dirs,files in os.walk(filepathOfBad):
        for file in files:
            realpath = os.path.join(root,file)

            with open(realpath,errors='ignore') as f:
                data.append((f.read(),'bad'))
    for root, dirs, files in os.walk(filepathOfGood):
        for file in files:
            realpath = os.path.join(root, file)

            with open(realpath, errors='ignore') as f:
                data.append((f.read(), 'good'))

    random.shuffle(data)
    return data


filepathOfBad = '/Users/huihui/PycharmProjects/ml_datamining/Ch04/mix20_rand700_tokens_cleaned/tokens/neg'
filepathOfGood = '/Users/huihui/PycharmProjects/ml_datamining/Ch04/mix20_rand700_tokens_cleaned/tokens/pos'



#print(res)

def train_and_test_dataSet(dataSet):
    filesize = int(0.85 * len(dataSet))

    train_data = [each[0] for each in dataSet[:filesize]]#dataSet=list(data,target)的形式
    train_target = [each[1] for each in dataSet[:filesize]]

    test_data = [each[0] for each in dataSet[filesize:]]
    test_target = [each[1] for each in dataSet[filesize:]]

    return train_data,train_target,test_data,test_target

dataSet = get_dataSet()
ss = dataSet[:2]
train_data, train_target, test_data, test_target = \
    train_and_test_dataSet(dataSet)



from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
def bayesClassifyTrainAndTest_MultinomialNB(train_data, train_target, test_data, test_target):
    nbc_6 = Pipeline(
        [
            ('vect',TfidfVectorizer(

            )),
            ('clf',MultinomialNB(alpha=3)),
        ]
    )

    nbc_6.fit(train_data,train_target)
    predict = nbc_6.predict(test_data)

    count = 0
    for left,right in zip(predict,test_target):
        if left == right:
            count += 1

    return (count / len(test_target))


def bayesClassifyTrainAndTest_BernoulliNB(train_data, train_target, test_data, test_target):
    nbc_1 = Pipeline([
        ('vect',TfidfVectorizer()),
        ('clf',BernoulliNB(alpha=3)),
    ])
    nbc_1.fit(train_data,train_target)
    predict = nbc_1.predict(test_data)

    count = 0
    for pred,targ in zip(predict,test_target):
        if pred == targ:
            count += 1
    return (count / len(test_target))

hh = bayesClassifyTrainAndTest_MultinomialNB(train_data, train_target, test_data, test_target)

print(hh)

jj = bayesClassifyTrainAndTest_BernoulliNB(train_data, train_target, test_data, test_target)
print(jj)


'''
两种算法下，结果差别不大

'''

















