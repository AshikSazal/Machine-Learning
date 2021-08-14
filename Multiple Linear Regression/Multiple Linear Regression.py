# Import Library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Import dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]


print(dataset.head())


# convert 'State' column to categorical column
states=pd.get_dummies(x['State'],drop_first=True)


# Drop the 'State' column
x = x.drop('State',axis=1)


# Concate the dummy variable
x = pd.concat([x,states],axis=1)

# Splitting the data into Training and Test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# Predict the test set
y_pred = regressor.predict(x_test)


print("R2 Score : ",metrics.r2_score(y_pred,y_test))
print("Intercept : ",regressor.intercept_)
print("Coefficients : ",list(zip(x, regressor.coef_)))
print("Error : ",metrics.mean_squared_error(y_pred,y_test))
