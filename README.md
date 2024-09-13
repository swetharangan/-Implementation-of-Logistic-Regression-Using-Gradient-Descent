# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize parameters: Set weights and bias to small random values or zeros.
2. Compute the prediction
3. Calculate the cost
4. Update parameters

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Preethi S
RegisterNumber:  212223230157
*/
```

## Output:

```
import pandas as pd
import numpy as np

dataset=pd.read_csv('Placement_Data.csv')
dataset
```
![image](https://github.com/user-attachments/assets/5935aabf-6064-4d6b-b049-9cf7b7309c7f)

```
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
```
![image](https://github.com/user-attachments/assets/ec9a5174-c9da-41a4-b2c4-f2f92ad06a4d)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/ed08bc42-22ec-4f21-848f-acea3649aee6)
```
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
```
![image](https://github.com/user-attachments/assets/c1d82e17-b5d1-4315-9715-1555b0e037e5)
```
#initialize the model paramenters
theta=np.random.randn(X.shape[1])
y=Y

#define sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#define loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)) + (1-y)*np.log(1-h)

#define gradient descent algorithm
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta


#train model
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)


#make prediction
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

#evaluate model
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy);
```

![image](https://github.com/user-attachments/assets/21cd3ee9-efd8-4ef6-bfa8-7c35238ded64)

```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/5f266d7b-0518-4c60-a251-fe0b2f896c8b)
```
print(Y)
```
![image](https://github.com/user-attachments/assets/cf690a6a-a102-45ac-8125-faa9f8ef10fd)
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/91d44de8-414d-4122-b06d-3c2e958a91aa)
```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/922a038d-0fce-4b9d-84e9-766b6bdf798b)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

