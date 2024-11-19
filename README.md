# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope  controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input
   Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data
   
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: swetha.R
RegisterNumber: 212223040221

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
print()
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
*/
```

## Output:

# Head and Tail

![Screenshot 2024-09-14 154846](https://github.com/user-attachments/assets/a9f319a1-c104-4c7f-aead-232905744783)

# X and Y

![Screenshot 2024-10-19 173916](https://github.com/user-attachments/assets/0b027f04-0a68-4934-92ab-a00e141f204a)

![Screenshot 2024-10-19 173929](https://github.com/user-attachments/assets/eb01b2e1-f76d-49f7-80e4-2c3462f4c11a)


# Training data

![Screenshot 2024-09-14 154930](https://github.com/user-attachments/assets/608786a8-45c5-4bf4-b022-9a34d9bde8f1)

# Plot for training set

![Screenshot 2024-09-14 154950](https://github.com/user-attachments/assets/5c5ae456-cdd0-4536-baec-20b0008bbfa6)

# Plot for test set

![Screenshot 2024-09-14 155012](https://github.com/user-attachments/assets/af3f6b1b-7bfd-4f6d-b6d3-f09a9e5528c8)

# MSE, MAE, RMSE values

![Screenshot 2024-09-14 155028](https://github.com/user-attachments/assets/807200e9-440c-4453-af3f-285d292451b9)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks 
