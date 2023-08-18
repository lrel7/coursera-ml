# Regularized logistic regression

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import utils

# g(z) = 1 / (1 + e^{-z})
def sigmoid(z):
    z = np.array(z) # convert input to a numpy array
    g = np.reciprocal(1 + np.exp(-z))
    return g

# read data
data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

def plotData(X, y):
    fig = plt.figure()
    pos = y == 1 # find indices of positive
    neg = y == 0 # find indices of negative
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
    plt.show()

plotData(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right')

# the above figure shows the dataset cannot be separated into positive and negative by a straight-line, therefore we need to map the features into all polynomial terms of x1 and x2 up to the sixth power
X = utils.mapFeature(X[:, 0], X[:, 1])

def costFunctionReg(theta, X, y, lambda_):
    m = y.size
    h = sigmoid(X @ theta)
    J = 1 / m * np.sum(-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)) + lambda_ / (2 * m) * np.sum(np.square(theta[1:]))
    grad = 1 / m * (h - y) @ X
    grad[1:] = grad[1:] + lambda_ / m * theta[1:] # should not regularize theta_0
    return J, grad

test_theta = np.ones(X.shape[1])
lambda_ = 10
cost, grad = costFunctionReg(test_theta, X, y, lambda_)
print('Cost at test_theta: {:.2f}'.format(cost))
print('Gradient at test_theta - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
