import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

dataset=pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Must include feature scaling because the SVR class does not do it automatically
sc_X = StandardScaler()
sc_y = StandardScaler()
X = np.reshape(X, (-1,1))
y = np.reshape(y, (-1,1))

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fit SVR to the dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Visualize the support vector regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("SVR")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
