# Normal equations

import os
import numpy as np
import utils

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta

theta = normalEqn(X, y)
print('Theta computed from the normal equation: {:s}'.format(str(theta)))
price = np.array([1, 1650, 3]) @ theta
print('Predicted price of a 1650 sq-ft, 3 bedrooms house: ${:.0f}'.format(price))
