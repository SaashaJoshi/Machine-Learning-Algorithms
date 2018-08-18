import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
dataSet = pd.read_csv('Salary_Data.csv')
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

# Splitting of data into training set and test set!
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest= train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()        #instantiating the LinearRegression model
regressor.fit(Xtrain, ytrain)       #fits model on training data
