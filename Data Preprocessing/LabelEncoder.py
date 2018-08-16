# Encoding categorical data

import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:, :-1].values     #taking all the rows of all the columns except the last one!
Y = dataSet.iloc[:, -1].values      #taking all the rows of the last column

labelX=LabelEncoder()
X[:,0]=labelX.fit_transform(X[:,0])
hotEncoder= OneHotEncoder(categorical_features=[0])   #OneHotEncoder!
X=hotEncoder.fit_transform(X).toarray()

labelY=LabelEncoder()
Y=labelY.fit_transform(Y)
