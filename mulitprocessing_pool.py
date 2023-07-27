import multiprocessing
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from stochastic_gradient2 import StochasticGradient


def run_sgd(args):
    # Unpack the arguments
    data, start, end, epochs, mini_batch_size = args

    # Run SGD on the subset of data
    data.sgd(epochs, mini_batch_size)

    return data.theta, data.errors


if __name__ == '__main__':
    num_ranks = 10
    processes = []

    start_time = time.time()

    # Calculate the size of each mini-batch
    mini_batch_size = len(StochasticGradient().target) // num_ranks

    # Create a list of arguments for each mini-batch
    args_list = []
    for i in range(num_ranks):
        data = StochasticGradient()
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size if i < num_ranks - \
            1 else len(data.target)
        args_list.append((data, start, end, 1000, mini_batch_size))

    # Use a process pool to process each mini-batch in parallel
    with Pool(num_ranks) as pool:
        results = pool.map(run_sgd, args_list)



    # Collect the theta values and the costs
    thetas = []
    errors = []
    for theta, error in results:
        thetas.append(theta)
        errors.append(error)

    end_time = time.time()

    print("Time Elapsed:", end_time - start_time, "seconds")

    # Merge all error lists into one
    merged_errors = [error for error_list in errors for error in error_list]

#  Plot the merged errors
    # plt.plot(range(len(merged_errors)), merged_errors)
    # plt.xlabel('Iteration')
    # plt.ylabel('Error')
    # plt.title('Error over iterations')
    # plt.show()# # Get the data
    X = np.array(range(len(merged_errors)))
    Y = np.array(merged_errors)
    X, Y = np.meshgrid(X, Y)
    Z = -(Y**2 + X ** 2)  # This should be your function of X and Y
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    ax.set_zticklabels([]) 
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Set the labels
  
    # Show the plot
    plt.show()
