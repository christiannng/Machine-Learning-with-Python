# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:07:05 2020

@author: Christian
"""

#Importing the libraries
import numpy as np #Mathematical library
import matplotlib.pyplot as plt #To plot nice chart
import pandas as pd #To import and manage dataset

#Import the dataset
dataset = pd.read_csv('Data.csv')
#Matrix of independent variables
X = dataset.iloc[:, :-1].values #Getting data from 3 columns in dataset
y = dataset.iloc[:, 3].values #Getting data from the 4th column (index start at 0)


#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
#We split test set and train set 80-20
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

#Feature Scaling: Standalization or Normalization
##In order to have independent variables in the same range
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""













