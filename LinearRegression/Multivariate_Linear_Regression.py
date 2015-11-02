__author__ = 'zhenxingzhang'
from numpy import loadtxt, zeros, ones, array, linspace, logspace, arange, mean, std
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np

#Plot the data
# scatter(input, output, marker='o', c='b')
# title('Profits distribution')
# xlabel('Population of City in 10,000s')
# ylabel('Profit in $10,000s')
# show()

def compute_cost(X, y, theta):
    '''
    :param X: the input data vector [m,d]
    :param y: the groundtruth output[m,1]
    :param theta: the weights [d+1, 1]
    :return: the squared error
    '''
    length = y.size

    predictions = X.dot(theta).flatten()
    sqErrors = (predictions - y) ** 2

    J = (1.0/(2 * length)) * sqErrors.sum()
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    :param X:
    :param y:
    :param theta:
    :param alpha:
    :param num_iters:
    :return:
    '''
    J_history = zeros(shape=(num_iters, 1))

    m = y.size
    for iter in range(num_iters):
        predictions = X.dot(theta)

        length = theta.size

        subtract_factor = np.zeros((length,1), dtype=np.float64)

        errors = predictions - y
        for i in range(length):
            error_sum = ( errors * X[:, i:i+1]).sum()
            subtract_factor[i][0] = alpha*(1.0/m)*error_sum

        theta = theta - subtract_factor

        J_history[iter] = compute_cost(X, y, theta)

    return theta, J_history


def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r

#Load the dataset
data = loadtxt('../Data/ex1data2.txt', delimiter=',')


#Plot the data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax.scatter(xs, ys, zs, c=c, marker=m)
ax.set_xlabel('Size of the House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of the House')
plt.show()

x = data[:, :2]
y = data[:, 2]

#number of training samples
m = y.size

y.shape = (m, 1)

#Scale features and set them to zero mean
x, mean_r, std_r = feature_normalize(x)

#Add a column of ones to X (interception data)
it = ones(shape=(m, 3))
it[:, 1:3] = x

#Some gradient descent settings
iterations = 100
alpha = 0.01

#Init Theta and Run Gradient Descent
theta = zeros(shape=(3, 1))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
print theta
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

predictions = it.dot(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = predictions.flatten()
    ax.scatter(xs, ys, zs, c=c, marker=m)
ax.set_xlabel('Size of the House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of the House')
plt.show()