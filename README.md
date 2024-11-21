# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather labeled data with input features X and Y for classification as 0 or 1
2. Set initial values for the model parameters (weight w and bias b), typicaly to small random values.
3. Use the sigmoid function to predict probabilities:
   P(y=1|X) = 1/(1+e^(wX+b))
4. Compute the binary cross_entropy loss to measure how well the model's predictions match the actual labels
5. Adjust w and b by computing their gradients with respect to the loss and applying gradient descent updates
6. Iterate over the training data, repeatedly updating the parameters until convergence (or for a fixed number of iterations)
7. Use the final model parameters to predict robabilities for new data and classify based on threshold (eg: 0.5 for binary calculation)
   
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Preethi S
RegisterNumber: 212223230157

import pandas as pd
import numpy as np
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h=sigmoid(X.dot(theta))
    retun -np.sum(y*np.log(h) + (1-y) * np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta, X, y, alpha = 0.01, num_iterations=1000)
def predict(theta, X):
    h=sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy: ",accuracy)
print(y_pred)
print(Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
*/
```

## Output:
Dataset

![Screenshot 2024-09-16 101916](https://github.com/user-attachments/assets/e27e523c-3db3-40df-99b5-a6e12dc13ac7)

![Screenshot 2024-09-16 101946](https://github.com/user-attachments/assets/f4548e3f-0723-458d-97e5-31adb191430a)

![Screenshot 2024-09-16 101956](https://github.com/user-attachments/assets/e432594c-7c64-4182-82cc-e041207cc62e)

# Array

![Screenshot 2024-09-16 102007](https://github.com/user-attachments/assets/f0d21389-20b1-48da-8fca-17908ee15fcd)

# Accuracy

![Screenshot 2024-09-16 102016](https://github.com/user-attachments/assets/80ccfbd3-ffc8-466d-ae52-6aa2365c4f48)

# Y Prediction

![Screenshot 2024-09-16 102023](https://github.com/user-attachments/assets/eb95a4d3-2ed9-443d-80c8-7f5e4c905f46)

# Y

![Screenshot 2024-09-16 102028](https://github.com/user-attachments/assets/3e27ea3f-c1a0-4ea6-a1fc-755cb48dbeca)

# New Prediction

![Screenshot 2024-09-16 102035](https://github.com/user-attachments/assets/d9b842e7-0ea5-4eab-af32-06aea7a36285)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
