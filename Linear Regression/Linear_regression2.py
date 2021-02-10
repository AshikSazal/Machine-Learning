# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values # Independent
y = dataset.iloc[:, 1].values   # Dependent

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred=regressor.predict(x_test)

# Visualizing the Training set results
plt.scatter(x_train,y_train,color='red',label='Real Salary')
plt.plot(x_train,regressor.predict(x_train),color='blue',label='Predicted Salary')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualizing the Test set results
plt.scatter(x_test,y_test,color='red',label='Real Salary')
plt.plot(x_train,regressor.predict(x_train),color='blue',label='Predicted Salary')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
