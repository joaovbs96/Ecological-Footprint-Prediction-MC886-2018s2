# Import
import ParseData
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# calculates gradient descent with regularization
def gradientDescentReg(x, y, alpha, n, m, it, reg):
    xTran = x.transpose()
    thetas = np.ones(n, dtype=float)
    J = np.zeros(it)

    for i in range(it):
        hypothesis = np.dot(x, thetas)
        diff = hypothesis - y
        J[i] = (np.sum(diff**2) + reg*np.sum(thetas**2))/(2*m)
        gradient = np.squeeze(np.dot(xTran, diff))/m
        thetas = np.squeeze(thetas - alpha * gradient)

        thetas = thetas * (1 - alpha * (reg / m)) - alpha * gradient

    return J, thetas

# function to calculate normal equation with regularization
def normalEquationReg(x, y, reg):
    identity = np.identity(x.shape[1])
    identity[0][0] = 0
    inverse = np.linalg.inv(np.dot(x.T, x) + reg*identity)
    thetas = np.dot(np.dot(inverse, x.T), y)

    return thetas

# function to calculate mean absolute error
def calcR2(x, y_true, theta):
    y_pred = np.dot(x, theta)

    return r2_score(y_true, y_pred)


## MAIN

# Get data set
x_train, y_train, x_valid, y_valid = ParseData.get_train_set()
x_test, y_test = ParseData.get_test_set()

# Scale data
# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# insert bias
x_train = np.insert(x_train, 0, 1, axis=1)
x_valid = np.insert(x_valid, 0, 1, axis=1)
x_test = np.insert(x_test, 0, 1, axis=1)

# calculate cost
m, n = x_train.shape
it = 10000
r = 10
alpha = 0.01

# execute GD with regularization
print('GD batch with regularization:')
J, thetas = gradientDescentReg(x_train, y_train, alpha, n, m, it, r)
print('Train: ' + str(calcR2(x_train, y_train, thetas)))
print('Validation: ' + str(calcR2(x_valid, y_valid, thetas)))
#print('Test: ' + str(calcR2(x_test, y_test, thetas)))
print()

# plot graph for GD with regularization
plt.plot(J, 'blue')
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('DG para alpha 0.1 e regularização 10')
plt.savefig('GDModel.png')
plt.gcf().clear()


# execute normal equation
print('NE with regularization:')
thetasNE = normalEquationReg(x_train, y_train, r)
print('Train: ' + str(calcR2(x_train, y_train, thetasNE)))
thetasNE = normalEquationReg(x_valid, y_valid, r)
print('Validation: ' + str(calcR2(x_valid, y_valid, thetasNE)))
#thetasNE = normalEquationReg(x_test, y_test, r)
#print('Test: ' + str(calcR2(x_test, y_test, thetasNE)))
print()
