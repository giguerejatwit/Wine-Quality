import multiprocessing as mp
import threading
import time
from multiprocessing import Process

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

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


# run a logistic regression
logit = LogisticRegression(random_state=0)
logit.fit(preprocessing.normalize(trainx), target)
# predict regression
pred = logit.predict(testx)

x0 = np.ones(trainx.shape[0])
x0t = np.ones(testx.shape[0])

trainx = np.c_[x0, trainx]
testx = np.c_[x0t, testx]



def logit(a):
    log = 1.0 / (1.0 + np.exp(-a))
    return log

def linear(x, theta):
    prod = np.dot(x, np.float64(theta))
    return prod

dim = trainx.shape[0]
theta = np.zeros(dim)
theta1 = np.zeros(dim)

theta = np.random.normal(dim)
theta1 = theta

# learning rate in the range[0,1]
alpha = 0.1
x = trainx[: , -13]


start_time = time.time()
for i in range(1, 1000):

     # loop over all of the data
    for j in range(len(x)):
        k = j
        xx = x[int(k)]
        yy = trainx[k, 12]
        a = linear(xx, theta)
        hh2 = logit(a)
        const = yy - hh2
        theta = theta + alpha * const * xx
    
    d = abs(theta1 - theta)
    err = np.max(d)
    

    if (err <= 1*10^-5):
        break
    else:
         print(err)
    theta1 = theta

print(theta)



end_time = time.time()



# Predict on the test data


# Print the coefficients and the mean squared error
# print("Coefficients: ", model.coef_)
# print("Mean squared error: ", mean_squared_error(test_data, y_pred))

# Print the results
print("Time taken:", end_time - start_time, "seconds")
