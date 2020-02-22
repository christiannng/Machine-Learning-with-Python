# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:11:25 2020

@author: Christian
"""

#polynomial_regression

#Importing the libraries
import numpy as np #Mathematical library
import matplotlib.pyplot as plt #To plot nice chart
import pandas as pd #To import and manage dataset

#Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#Matrix of independent variables
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values 


#Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
#We split test set and train set 80-20
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 0)"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualising the linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regresssion)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the polynomial results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regresssion)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)

#Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))                                                   


