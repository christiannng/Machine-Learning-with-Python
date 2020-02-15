# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:46:23 2020

@author: Christian
"""

#Simple Linear Regression
#Importing the libraries
import numpy as np #Mathematical library
import matplotlib.pyplot as plt #To plot nice chart
import pandas as pd #To import and manage dataset

#Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
#Matrix of independent variables
X = dataset.iloc[:, :-1].values #Getting data from 3 columns in dataset
y = dataset.iloc[:, 1].values #Getting data from the 4th column (index start at 0)


#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
#We split test set and train set 80-20
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 1/3,
                                                    random_state = 0)

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the test set results
y_pred = regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()




