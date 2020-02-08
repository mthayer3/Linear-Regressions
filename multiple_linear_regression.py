# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler




dataset =pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values #Independent 
y = dataset.iloc[:, 4].values #Dependent


#Convert categorical variables
labelencoder_X = LabelEncoder()
#labelenconder fit_transform transforms the categories strings into integers
X[:, 3] = labelencoder_X.fit_transform(X[: , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
#onehotencoder fit_transform creates separate columns for each different labels. The separate columns are the dummy variables
X = onehotencoder.fit_transform(X).toarray()

#Avoid the Dummy Variable Trap
#Not neccessary because the python libraries will do this for you to but it is good practice
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
