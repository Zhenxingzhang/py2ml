from numpy import loadtxt, zeros, ones, array, linspace, logspace, arange
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
    for iter in range(num_iters-1):
        predictions = X.dot(theta[:, iter]).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][iter+1] = theta[0][iter] - alpha*(1.0/m)*errors_x1.sum()
        theta[1][iter+1] = theta[1][iter] - alpha*(1.0/m)*errors_x2.sum()

        J_history[iter] = compute_cost(X, y, theta[:, iter])

    return theta, J_history

theta = zeros(shape=(2, 1))

m = input.size
#Add a column of ones to X (interception data)
X = ones(shape=(m, 2))
X[:, 1] = input

print compute_cost(X, output, theta)

#Some gradient descent settings
iterations = 100
alpha = 0.01

theta = zeros(shape=(2, iterations))

theta, J_h = gradient_descent(X, output, theta, alpha, iterations)

print J_h

#Plot the results
result = X.dot(theta[:,iterations-1]).flatten()
plot(data[:, 0], result)
show()

#Plot the convergence of cost function
plot(arange(iterations), J_h)
xlabel('Iterations')
ylabel('Cost Function')
show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

print theta[0,:].shape, theta[1,:].shape, J_h.shape

ax.plot(theta[0,:].flatten(), theta[1,:].flatten(), J_h.flatten())
ax.set_xlabel("Theta 0 Axis")
ax.set_ylabel("Theta 1 Axis")
ax.set_zlabel("Cost Function Axis")
ax.set_title("Overview")

plt.show()


#Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)


#initialize J_vals to a matrix of 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(X, output, thetaT)

#Contour plot
J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
show()


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals[:, 1], rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()