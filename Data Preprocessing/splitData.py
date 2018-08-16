# The code splits data into training set and test set

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:, :-1].values     #taking all the rows of all the columns except the last one!
Y = dataSet.iloc[:, -1].values   

labelX=LabelEncoder()
X[:,0]=labelX.fit_transform(X[:,0])
hotEncoder= OneHotEncoder(categorical_features=[0])
X=hotEncoder.fit_transform(X).toarray()

labelY=LabelEncoder()
Y=labelY.fit_transform(Y)

Xtrain, Xtest, Ytrain, Ytest= train_test_split(X, Y, test_size=0.2, random_state=0)
