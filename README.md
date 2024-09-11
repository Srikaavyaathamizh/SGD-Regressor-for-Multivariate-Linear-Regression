![image](https://github.com/user-attachments/assets/87814606-1d40-485c-9e3b-3c9a9e76358c)# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start Step
2.Data Preparation
3.Hypothesis Definition
4.Cost Function 
5.Parameter Update Rule 
6.Iterative Training 
7.Model Evaluation 
8.End
``

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SRIKAAVYAA T
RegisterNumber:  212223230214

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
*/
```
![Screenshot 2024-09-04 134115](https://github.com/user-attachments/assets/7352d2cb-4077-432d-8a02-c2eef1ce7f57)

```
df.info()
```
![Screenshot 2024-09-11 132140](https://github.com/user-attachments/assets/a4aa89d1-f21b-4990-b961-ac45a03599b5)
```
X=df.drop(columns=['AveOccup','target'])
X.info()
```
![Screenshot 2024-09-11 132230](https://github.com/user-attachments/assets/6ce45980-844a-4b00-9da8-6923f9ff54d2)

```
Y=df[['AveOccup','target']]
Y.info()
```
![Screenshot 2024-09-11 132457](https://github.com/user-attachments/assets/9f7d87f0-8df1-4475-a5c5-0e49e85e51e6)

```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
```
![Screenshot 2024-09-11 132603](https://github.com/user-attachments/assets/2592faf5-8e18-43a8-b077-a6756cb836a3)

```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
```
![Screenshot 2024-09-11 132712](https://github.com/user-attachments/assets/aa070236-45a5-4d3e-a2f7-2e36cbc71559)
```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])

```

## Output:
![Screenshot 2024-09-11 132826](https://github.com/user-attachments/assets/6acbf44b-128e-42df-9643-f8639f235f79)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
