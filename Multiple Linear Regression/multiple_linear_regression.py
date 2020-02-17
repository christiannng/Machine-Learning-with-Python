# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:30:01 2020

@author: Christian
"""

#Multiple Linear Regression

#Importing the libraries
import numpy as np #Mathematical library
import matplotlib.pyplot as plt #To plot nice chart
import pandas as pd #To import and manage dataset

#Import the dataset
dataset = pd.read_csv('50_Startups.csv')
#Matrix of independent variables: X = independent vars; y = dependent vars
X = dataset.iloc[:, :-1].values #Getting data from 3 columns in dataset
y = dataset.iloc[:, 4].values #Getting data from the 4th column (index start at 0)

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##Encoding independent variables
labelencoder_X = LabelEncoder()
#However, this could cause the problem where our algo would think that 
##French has higher value than Germany, etc.
##To solve this, we would have 3 columns represent 0 and 1
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) 
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
#Usually the library do this for uus
X = X[:, 1:]


#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
#We split test set and train set 80-20
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

#Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)







