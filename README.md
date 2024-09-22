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
```
## PROGRAM:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SRIKAAVYAA T
RegisterNumber: 212223230214

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
```

![Screenshot 2024-09-22 231927](https://github.com/user-attachments/assets/b689b6f3-e44b-4c59-8e3f-db438219f3ab)

```
df.info()
```
![Screenshot 2024-09-22 232252](https://github.com/user-attachments/assets/61e8659c-8c4c-495e-9127-e11de2b5b17e)

```
X=df.drop(columns=['AveOccup','target'])
X.info()
```
![Screenshot 2024-09-22 232426](https://github.com/user-attachments/assets/3ac445f1-378d-4276-8cf1-d2f977b28609)

```

Y=df[['AveOccup','target']]
Y.info()
```
![Screenshot 2024-09-22 232858](https://github.com/user-attachments/assets/52ff0791-b3c3-4f4f-b3bf-2a1a9762ab5c)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
```
![Screenshot 2024-09-22 233005](https://github.com/user-attachments/assets/7f600777-79b6-4deb-94d6-52b91edbe2e1)

```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
```
![Screenshot 2024-09-22 233147](https://github.com/user-attachments/assets/522a1b00-b552-4316-affc-c6ba80f3e5c7)
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
![Screenshot 2024-09-22 233356](https://github.com/user-attachments/assets/247c41f9-5f3b-4658-8733-77cb4907f0b7)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
