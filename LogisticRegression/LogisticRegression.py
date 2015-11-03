import math, numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

def sigmoid(x):
    return 1/(1+ math.exp(-x))

def predict(x, theta):
    return map(sigmoid, x.dot(theta))

def costFunction(X, y, theta):

    m = y.size
    predictions = predict(X, theta)

    subtraction = np.ones((y.size))

    cost_rows = -1 *(y * map(math.log, predictions)) - (subtraction-y).flatten()* map(math.log, (subtraction-predictions).flatten())
    return cost_rows.sum()/m

def gradient_descent(X, y, theta, alpha, iters):

    m = 3
    J_history = np.zeros((iters ,1))

    for iter in range(iters):

        predictions = predict(X, theta)

        subtract_factor = np.zeros((m,1), dtype=np.float64)

        errors = predictions - y

        for i in range(m):

            error_sum = ( errors * X[:, i]).sum()
            subtract_factor[i][0] = alpha*(1.0/m)*error_sum

        theta = theta - subtract_factor
        # print theta

        J_history[iter, 0] = costFunction(X, y, theta)

    return theta, J_history

# #test sigmoid function
# input = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
# result = map(sigmoid, input)
# plt.plot(input, result)
# plt.ylabel('Sigmoid')
# plt.show()


#load the dataset
data = np.loadtxt('../Data/ex2data1.txt', delimiter=',')

X = data[:, :2]/100
y = data[:, 2]
print y.shape
# #plot the training dataset
# pos_indices = [i for i, x in enumerate(y) if x == 1]
# neg_indices = [i for i, x in enumerate(y) if x == 0]
#
# plt.plot(X[pos_indices, 0], X[pos_indices, 1], 'k+')
# plt.plot(X[neg_indices, 0], X[neg_indices, 1], 'yo')
# plt.show()

#test predict function
theta = np.zeros((3,1))

input = np.ones((y.size,3))
input[:,1:3] = X[:,:2]

theta = [[0.03333333],[4.0030722],[ 3.75428074]]

# print input.dot(theta)
# print input
# prediction = predict(input, theta)
# print input
# print prediction
# print input
# cost = costFunction(input, y, theta)
# print input
# print cost

#test gradient_descent
theta = np.zeros((3,1), dtype=np.float)
iters = 2000
alpha = 0.05

theta_r, j_history = gradient_descent(input, y , theta, alpha, iters)
print theta_r

plot(np.arange(iters), j_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

#plot the training dataset
pos_indices = [i for i, x in enumerate(y) if x == 1]
neg_indices = [i for i, x in enumerate(y) if x == 0]

print theta_r[0], theta_r[1], theta_r[2]
x1 = theta_r[2]/-theta_r[0]
y1 = theta_r[1]/-theta_r[0]
print x1, y1
plt.plot(X[pos_indices, 0], X[pos_indices, 1], 'k+')
plt.plot(X[neg_indices, 0], X[neg_indices, 1], 'yo')
plt.plot([0, x1], [y1, 0], 'ro')
plt.show()