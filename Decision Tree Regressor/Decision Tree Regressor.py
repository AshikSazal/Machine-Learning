# Import the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


# Decision Tree model
regressor = DecisionTreeRegressor(random_state=0)


# Fit the regressor model to the dataset
regressor.fit(x,y)


# Prediction
y_pred = regressor.predict([[5.5]])
print(y_pred)


# Visualization the results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()