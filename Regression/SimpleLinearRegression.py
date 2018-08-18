# The machine in this machine learning model is the Simple Linear Regression Model and learning means that we trained SLR machine model on the training set.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
dataSet = pd.read_csv('Salary_Data.csv')
X = dataSet.iloc[:, :-1].values     # independent variable
y = dataSet.iloc[:, -1].values      # dependent variable

# Splitting of data into training set and test set!
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest= train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()        # instantiating the LinearRegression model
regressor.fit(Xtrain, ytrain)       # fits model on training data

# Predicting test results
ypred=regressor.predict(Xtest)

# Plotting the training set
plt.scatter(Xtrain, ytrain)     # plotting the variables of the training set
plt.plot(Xtrain, regressor.predict(Xtrain), color='Black')    # regression line!
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Plotting the test set
plt.scatter(Xtest, ytest)     # plotting the variables of the test set
plt.plot(Xtrain, regressor.predict(Xtrain), color='Black')      # Regression line of the already trained machine model (using the training set)
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
  
