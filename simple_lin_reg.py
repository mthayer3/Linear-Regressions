# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:35:34 2020

@author: matth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset =pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, 0].values #Independent 
y = dataset.iloc[:, 1].values #Dependent
X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Linear Regression
#Create a regressor object from the LinearRegression class
regressor = LinearRegression()

#Use fit method to fit regressor to our dataset
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Visualize the training set
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs. Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Visualize the test set
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs. Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()



