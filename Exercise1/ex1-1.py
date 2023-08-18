# Gradient descent

import os
import numpy as np
import utils

# read data
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter = ',')
X, y = data[:, :2], data[:, 2]
m = y.size # number of training examples

# feature normalization
# (when features differ by orders of magnitude, first perfoming feature scaling can make gradient descent converge much more quickly)
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X)
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# add the intercept term
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

def computeCost(X, y, theta):
    m = y.size
    J = 0.5 / m * np.sum(np.square(X @ theta - y))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    theta = theta.copy()
    J_history = []

    for i in range(num_iters):
        theta = theta + alpha / m * ((y - X @ theta).T @ X)
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

# gradient descent
alpha = 0.1
num_iters = 400
theta = np.zeros(3)
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
print('theta computed from gradient descent: {:s}'.format(str(theta)))

# predict the price of a 1650 sq-ft, 3 bedrooms house
# (need to normalize first)
price = np.insert(((np.array([1650, 3]) - mu) / sigma), 0, 1).T @ theta
print('Predicted price of a 1650 sq-ft 3 bedrooms house: ${:.0f}'.format(price))
