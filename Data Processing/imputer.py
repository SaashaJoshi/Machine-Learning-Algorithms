# The code replaces missing data with appropriate info/data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer

dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:, :-1].values     #taking all the rows of all the columns except the last one!
Y = dataSet.iloc[:, -1].values      #taking all the rows of the last column

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)    #axis=0 is columns
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

''' or
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
'''
