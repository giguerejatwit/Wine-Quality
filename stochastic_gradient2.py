import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


class StochasticGradient():

    wine_data = pd.read_csv("winequality_red.csv", delimiter=",")
    # 1599 rows of data
    num_rows = wine_data.shape[0]
    # 12 columns of data
    num_cols = wine_data.shape[1]

    # learning rate in the range[0,1]
    alpha = 0.001

    costs = []
    errors = []
    thetas = []

    def __init__(self):

        # sampling data -30/70 split
        self.ratio = int(self.num_rows * (0.7))
        self.idx = np.random.choice(
            range(self.num_rows), self.ratio, replace=False)

        # training data
        self.trainx = np.array(self.wine_data.iloc[self.idx])
        # testing data
        self.testx = np.array(self.wine_data.iloc[-self.idx])

        # transform data using value - mean / std
        for i in range(self.num_cols):
            self.trainx[:, i] = self.trainx[:, i] - \
                (np.mean(self.trainx[:, i]) / np.std(self.trainx[:, i]))

        for i in range(self.num_cols):
            self.testx[:, i] = self.testx[:, i] - \
                (np.mean(self.testx[:, i]) / np.std(self.testx[:, i]))

        # target variable -'Quality'
        self.target = self.wine_data.iloc[self.idx, -1].tolist()

        # self.model a logisitic regression
        self.model = LogisticRegression(random_state=0)
        self.model.fit(preprocessing.normalize(self.trainx), self.target)

        # predict regression
        self.pred = self.model.predict(self.testx)

        # add a column of 1's
        self.x0 = np.ones(self.trainx.shape[0])
        self.x0t = np.ones(self.testx.shape[0])
        # print(self.trainx[:])
        self.trainx = np.c_[self.x0, self.trainx]
        self.testx = np.c_[self.x0t, self.testx]

        dim = self.trainx.shape[0]
        self.theta = np.zeros(dim)
        self.theta1 = np.zeros(dim)

        self.theta = np.random.normal(dim)
        self.theta1 = self.theta
        # x is the updated target vector
        self.x = self.trainx[:, -13]
        # print(self.x)

    def logit(self, a):
        # Clip a to avoid overflow
        a = np.clip(a, -500, 500)
        log = 1.0 / (1.0 + np.exp(-a))
        return log

    def linear(self, x, theta):
        prod = np.dot(x, np.float64(theta))
        return prod

    def cost_function(self, X, y, theta):
        m = len(y)
        h = self.logit(np.dot(X, theta))
        y_array = np.array(y)  # Convert list to numpy array
        epsilon = 1e-7  # Small constant to prevent log(0)
        cost = (-1 / m) * np.sum(y_array.reshape(-1, 1) * np.log(h +
                                                                 epsilon) + (1 - y_array.reshape(-1, 1)) * np.log(1 - h + epsilon))
        return cost

    def gradient(self, X, y, theta):
        m = len(y)
        h = self.logit(np.dot(X, theta))
        grad = (1 / m) * np.dot(X.T, (h - y))
        return grad

    def sgd(self, epochs=1000, mini_batch_size=100):
        gradient = []
        num_batches = len(self.x) // mini_batch_size

       # Convert self.target to a numpy array
        target_array = np.array(self.target)

        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            indices = np.random.permutation(len(self.x))
            x_shuffled = self.x[indices]
            y_shuffled = target_array[indices]  # Use the numpy array here

            # Divide the data into mini-batches and process each mini-batch
            for i in range(num_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                indices = np.random.permutation(len(self.x))
                x_shuffled = self.x[indices]
                y_shuffled = target_array[indices]
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Here you would update the model parameters based on x_batch and y_batch

                # Calculate the gradient
                gradient = self.gradient(x_batch, y_batch, self.theta)

                # Update the theta values
                self.theta = self.theta - self.alpha * gradient

                # After updating the theta values, calculate the cost (error)
                cost = self.cost_function(x_batch, y_batch, self.theta)
            self.errors.append(cost)


# ------ Plot Results -------

# Plot the cost over iterations
class Plot():

    def show(self):
        plt.show()

    def linearPlot(self):

        fig = plt.figure()
        ax = plt.axes()

        ax.plot(self.X, self.Y)

    def quadraticPlot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')   # Create the axes

        # Data
        self.X, self.Y = np.meshgrid(self.X, self.Y)

        Z = self.X**2 + self.Y**2
        # Plot the 3d surface
        surface = ax.plot_surface(self.X, self.Y, Z,
                                  cmap=cm.coolwarm,
                                  rstride=1,
                                  cstride=1)

        # Set some labels
        ax.set_xlabel('x-axis: Thetha')
        ax.set_ylabel('y-axis: Error Correction')
        ax.set_zlabel('Error: X^2 + Y^2')

        plt.show()

    def __init__(self, data) -> None:

        self.Y = data.errors
        self.X = range(1, 1000)
