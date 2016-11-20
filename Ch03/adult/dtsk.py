import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

# read data from excel file as DataFrame
raw_train_data = pd.read_excel("/Users/boyuan/Desktop/TrainingData.xlsx", parse_cols=[1,2,3,4,5,6,7,8,9,10,11])
raw_test_data = pd.read_excel("/Users/boyuan/Desktop/TestingData.xlsx", parse_cols=[1,2,3,4,5,6,7,8,9,10,11])

# If the data has missing values, they will become NaNs in the resulting Numpy arrays.
# The vectorizer will create additional column <feature>=NA for each feature with NAs

raw_train_data = raw_train_data.fillna("NA")
raw_test_data = raw_test_data.fillna("NA")

exc_cols = [u'adjGross']
cols = [c for c in raw_train_data.columns if c not in exc_cols]

X_train = raw_train_data.ix[:,cols]
y_train = raw_train_data['adjGross'].values

X_test = raw_test_data.ix[:,cols]
y_test = raw_test_data['adjGross'].values

# Convert DataFrame to dict See more: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.to_dict.html#pandas.DataFrame.to_dict
dict_X_train = X_train.to_dict(orient='records')
dict_X_test = X_test.to_dict(orient='records')

vec = DictVectorizer()
X_train = vec.fit_transform(dict_X_train).toarray()
X_test = vec.fit_transform(dict_X_test).toarray()

