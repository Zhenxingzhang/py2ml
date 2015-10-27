from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np


#Load the dataset
data = loadtxt('../Data/ex1data1.txt', delimiter=',')

input = data[:, 0]
output = data[:, 1]

#Plot the data
scatter(input, output, marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
show()

def compute_cost(X, y, theta):
    '''
    :param X: the input data vector [m,d]
    :param y: the groundtruth output[m,1]
    :param theta: the weights [d+1, 1]
    :return: the squared error
    '''
    m = y.size

    predictions = X.dot(theta)
    sqErrors = (predictions - y) ** 2

    J = (1.0/(2 * m)) * sqErrors.sum()

theta = [0.5 , 1]

m = input.size
print input
print ones((m,1))

X = np.append( ones((m,1)), input, 1)

print X