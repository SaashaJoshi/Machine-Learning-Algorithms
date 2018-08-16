# Dummy encoding categorical data

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:, :-1].values     #taking all the rows of all the columns except the last one!
Y = dataSet.iloc[:, -1].values   

labelX=LabelEncoder()
X[:,0]=labelX.fit_transform(X[:,0])
hotEncoder= OneHotEncoder(categorical_features=[0])
X=hotEncoder.fit_transform(X).toarray()
