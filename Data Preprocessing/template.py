import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:, :-1].values    
Y = dataSet.iloc[:, -1].values   

# Splitting of data into training set and test set!
from sklearn.model_selection import train_test_split      
Xtrain, Xtest, Ytrain, Ytest= train_test_split(X, Y, test_size=0.2, random_state=0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
Xtrain=scX.fit_transform(Xtrain)
Xtest=scX.transform(Xtest)
'''
