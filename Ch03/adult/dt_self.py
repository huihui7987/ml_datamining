import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

# read data from excel file as DataFrame
raw_train_data = pd.read_csv("/Users/huihui/PycharmProjects/machinelearninginaction/Ch03/lenses.txt")
#raw_test_data = pd.read_csv("/Users/huihui/PycharmProjects/machinelearninginaction/Ch03/lenses.txt")

# If the data has missing values, they will become NaNs in the resulting Numpy arrays.
# The vectorizer will create additional column <feature>=NA for each feature with NAs

raw_train_data = raw_train_data.fillna("NA")
#raw_test_data = raw_test_data.fillna("NA")

exc_cols = ['resclass']
cols = [c for c in raw_train_data.columns if c not in exc_cols]#前14列属性，最后一列为分类

X_train = raw_train_data.ix[:,cols]#前14列数据
y_train = raw_train_data['resclass'].values#最后一列数据

#X_test = raw_test_data.ix[:,cols]
#y_test = raw_test_data['resclass'].values
#
# Convert DataFrame to dict See more: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.to_dict.html#pandas.DataFrame.to_dict
dict_X_train = X_train.to_dict(orient='records')
#dict_X_test = X_test.to_dict(orient='records')

vec = DictVectorizer()
X_train = vec.fit_transform(dict_X_train).toarray()
#X_test = vec.fit_transform(dict_X_test).toarray()

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train,y_train)

#score = clf.score(X_test,y_test)

from sklearn.externals.six import StringIO


with open("len.dot", 'w') as f:
  f = tree.export_graphviz(clf, out_file=f, feature_names= vec.get_feature_names())




