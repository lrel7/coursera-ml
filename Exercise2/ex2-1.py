# Logistic regression

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import utils

# read data
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

def plotData(X, y):
    fig = plt.figure()
    pos = y == 1 # find indices of positive
    neg = y == 0 # find indices of negative
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
    plt.show()

# display the loaded data
plotData(X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'])

# g(z) = 1 / (1 + e^{-z})
def sigmoid(z):
    z = np.array(z) # convert input to a numpy array
    g = np.reciprocal(1 + np.exp(-z))
    return g

m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1) # add intercept term

def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X @ theta) # h = g(X @ theta)
    J = 1 / m * np.sum(-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h))
    grad = 1 / m * (h - y) @ X
    return J, grad

options = {'maxfun': 400}
initial_theta = np.zeros(n+1)
res = optimize.minimize(costFunction, initial_theta, (X, y), jac=True, method='TNC', options=options)
cost = res.fun
theta = res.x
print('Cost at theta: {:.3f}'.format(cost))
print('theta: {:.3f}, {:.3f}, {:.3f}'.format(*theta))

# produce '1' or '0' predictions given a dataset
def predict(theta, X):
    h = sigmoid(X @ theta)
    pos = h >= 0.5 # computes the predictions for X using a threshold at 0.5
    p = np.zeros(m)
    p[pos] = 1
    p[~pos] = 0
    return p

# predict probability for a student with score 45 on exam 1 and score 85 at exam 2
prob = sigmoid(np.array([1, 45, 85]) @ theta)
print('For a student with score 45 and 85, we predict an admission probability of {:.3f}'.format(prob))

# compute accuracy on our training set
p = predict(theta, X)
print('Train accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
