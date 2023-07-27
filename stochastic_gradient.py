import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def logit(a):
    log = 1.0 / (1.0 + np.exp(-a))
    return log

def linear(x, theta):
    prod = np.dot(x, np.float64(theta))
    return prod

def cost_function(X, y, theta):
    m = len(y)
    h = logit(np.dot(X, theta))
    cost = (-1 / m) * np.sum(y.reshape(-1, 1) * np.log(h) + (1 - y.reshape(-1, 1)) * np.log(1 - h))
    return cost

wine_data = pd.read_csv("winequality_red.csv", delimiter=",")

# 1599 rows of data
num_rows = wine_data.shape[0]
# 12 columns of data
num_cols = wine_data.shape[1]

# sampling data -30/70 split
ratio = int(num_rows * (0.7))
idx = np.random.choice(range(num_rows), ratio, replace=False)

# training data
trainx = np.array(wine_data.iloc[idx])
# testing data
testx = np.array(wine_data.iloc[-idx])

# transform data using value - mean / std
for i in range(num_cols):
    trainx[:, i] = trainx[:, i] - (np.mean(trainx[:, i]) / np.std(trainx[:, i]))

for i in range(num_cols):
    testx[:, i] = testx[:, i] - (np.mean(testx[:, i]) / np.std(testx[:, i]))


# target variable -'Quality'
target = wine_data.iloc[idx, -1]


# model a logisitic regression
model = LogisticRegression(random_state=0)
model.fit(preprocessing.normalize(trainx), target)

# predict regression
pred = model.predict(testx)

# add a column of 1's
x0 = np.ones(trainx.shape[0])
x0t = np.ones(testx.shape[0])
trainx = np.c_[x0, trainx]
testx = np.c_[x0t, testx]

dim = trainx.shape[0]
theta = np.zeros(dim)
theta1 = np.zeros(dim)

theta = np.random.normal(dim)
theta1 = theta

# learning rate in the range[0,1]
alpha = 0.1
x = trainx[: , -13]


start_time = time.time()

costs = []
errors = []
thetas = []

for i in range(1, 1000):

    # loop over all of the data
    for j in range(len(x)):
        xx = x[int(j)]
        yy = trainx[j, 12]
        a = linear(xx, theta)
        
        # gradient function
        hh2 = logit(a)
        
        # cost function
        const = yy - hh2
        theta = theta + alpha * const * xx
    
    cost = cost_function(trainx, x, theta)
    costs.append(cost)
    
    
    thetas.append(theta)

    d = abs(theta1 - theta)
    err = np.max(d)
    errors.append(err)

    if (err <= 1e-5):
        break
    else:
        print(err)
    theta1 = theta

# print(theta)

end_time = time.time()

# Print the results
print("Time taken:", end_time - start_time, "seconds")


# ------ Plot Results --------

# Plot the cost over iterations
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')   # Create the axes
# Data
# X = thetas


# Y = errors
# X, Y = np.meshgrid(X, Y)
# Z = X**2 + Y**2
# 
# Plot the 3d surface
# surface = ax.plot_surface(X, Y, Z,
                        #   cmap=cm.coolwarm,
                        #   rstride = 1,
                        #   cstride = 1)
# 
# Set some labels
# ax.set_xlabel('x-axis: Thetha')
# ax.set_ylabel('y-axis: Error Correction')
# ax.set_zlabel('Error: X^2 + Y^2')
# 
# plt.show()