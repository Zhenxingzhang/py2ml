from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np


#Load the dataset
data = loadtxt('../Data/ex1data1.txt', delimiter=',')

input = data[:, 0]
print input.shape
print type(input)
output = data[:, 1]
print input.shape

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

theta = [0.5 , 1]

m = input.size
#Add a column of ones to X (interception data)
X = ones(shape=(m, 2))
X[:, 1] = input

print compute_cost(X, output, theta)