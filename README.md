# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the dataset from the CSV file and extract the input feature (R&D Spend) and output variable (Profit). Apply feature scaling to the input data.

2.Initialize parameters such as weight w, bias b, learning rate α, and number of iterations (epochs).

3.Perform Gradient Descent:

 Predict output using the equation ŷ = w·x + b

 Compute the Mean Squared Error (loss)

 Calculate gradients dw and db

 Update parameters w and b

4.Repeat the process for the given number of iterations and display the final regression line and loss curve.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SAKTHI SABARISH P
RegisterNumber:  212225040360
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses = []
for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y)**2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)
    
    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y)
x_sorted = np.argsort(x)
plt.plot(x[x_sorted],(w * x + b)[x_sorted], color='red')
plt.xlabel("R&D Spend(scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()
print("Final Weight (w):", w)
print("Final Bias (b):", b)
```

## Output:
![Screenshot_30-1-2026_142627_localhost](https://github.com/user-attachments/assets/b85a275f-96d9-4837-914b-3ae41cf943cc)

```
Final Weight (w): 33671.51979690389
Final Bias (b): 97157.57273469678

```



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
