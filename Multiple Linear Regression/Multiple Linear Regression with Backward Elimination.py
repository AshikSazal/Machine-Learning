# Import the Library

import numpy as np
import pandas as pd
from sklearn import *

# Import the dataset
dataset=pd.read_csv("50_CompList.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values


# Encoding dummy variables : categorical feature convert to numerical value
ct = compose.ColumnTransformer([("State", preprocessing.OneHotEncoder(), [3])],remainder = 'passthrough')
x = ct.fit_transform(x)
print(x)


# As we can see in the above output, the state column has been converted into dummy variables (0 and 1).The first column corresponds to the California State, the second column corresponds to the Florida State, and the third column corresponds to the New York State.

# Note: We should not use all the dummy variables at the same time, so it must be 1 less than the total number of dummy variables, else it will create a dummy variable trap.

# Now, we are writing a single line of code just to avoid the dummy variable trap. Remove the first column
x = x[:,1:]


# Split the data into train & test set
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=.2,random_state=0)


# Create & train the model
regressor = linear_model.LinearRegression()
regressor.fit(x_train,y_train)


# Prediction
y_pred = regressor.predict(x_test)


# Get some score
print("R2 Score : ",metrics.r2_score(y_pred,y_test))
print("Intercept : ",regressor.intercept_)
print("Coefficients : ",list(zip(x, regressor.coef_)))
print("Error : ",metrics.mean_squared_error(y_pred,y_test))


# Get the train & test score
print('Train Score: ', regressor.score(x_train, y_train))
print('Test Score: ', regressor.score(x_test, y_test))  


# Backward Elimination
# (https://www.javatpoint.com/backward-elimination-in-machine-learning)

# Backward elimination is a feature selection technique while building a machine learning model. It is used to remove those features that do not have a significant effect on the dependent variable or prediction of output.
"""
Steps of Backward Elimination
 
Below are some main steps which are used to apply backward elimination process:
 
Step-1: Firstly, We need to select a significance level to stay in the model. (SL=0.05)
 
Step-2: Fit the complete model with all possible predictors/independent variables.
 
Step-3: Choose the predictor which has the highest P-value, such that.
 
    a. If P-value >SL, go to step 4.
    b. Else Finish, and Our model is ready.
 
Step-4: Remove that predictor.
 
Step-5: Rebuild and fit the model with the remaining variables.
"""

# Step-1:

# Import the library
import statsmodels.api as sm


# Adding a extra feature b0 which is constant term. So, we need to add a column x0=1 that will be helpful
x = np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
print(x.shape)


# Step-2: Now, we are actually going to apply a backward elimination process.

# We have to use all possible combination of independent features that are significantly affecting the dependent variable.
x_opt=np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt=np.array(x[:, [0,2,3,4,5]],dtype=float)
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()


x_opt= np.array(x[:, [0,3,4,5]],dtype=float)  
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()


x_opt=np.array(x[:, [0,3,5]],dtype=float)
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()


x_opt=np.array(x[:, [0,3]],dtype=float)
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()


# After remove some feature, above summary we can see that p value is low means 0. Now we can use this feature for next step
print(x_opt)


# ### Below is the code for Building Multiple Linear Regression model by only using R&D spend:

# In[18]:


# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd   
  
# Import the Independent and dependent Variable
x_BE=x_opt[:,1].reshape(-1,1)
y_BE=y
  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)  
  
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(nm.array(x_BE_train).reshape(-1,1), y_BE_train)  
  
#Predicting the Test set result;  
y_BE_pred = regressor.predict(x_BE_test)  


print("R2 Score : ",metrics.r2_score(y_BE_pred,y_BE_test))
print("Intercept : ",regressor.intercept_)
print("Coefficients : ",list(zip(x, regressor.coef_)))
print("Error : ",metrics.mean_squared_error(y_BE_pred,y_BE_test))
  
#Cheking the score  
print('Train Score: ', regressor.score(x_BE_train, y_BE_train))  
print('Test Score: ', regressor.score(x_BE_test, y_BE_test))
