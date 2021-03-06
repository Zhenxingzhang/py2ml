import math, numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

def predict(x, theta):
    return map(sigmoid, x.dot(theta))

def sigmoid(x):
    return long(1.0) /(1.0 + math.exp(-1.0 * x))

def costFunction(X, y, theta):

    m = y.size
    predictions = predict(X, theta)
    subtraction = np.ones((y.size), dtype=np.float)

    cost_rows = -1.0 * (y * map(math.log, predictions)) - (subtraction-y) * map(math.log, (subtraction-predictions))
    return cost_rows.sum()/m

def gradient_descent(X, y, theta, alpha, iters):

    m = y.size
    J_history = np.zeros((iters ,1))

    for iter in range(iters):

        predictions = predict(X, theta)

        size = theta.size

        errors = predictions - y

        # subtract_factor = np.zeros((3, 1))

        subtract_factor = np.array([(alpha * (1.0/m) * np.sum(errors.T * X.T, axis=1))]).T
        # for i in range(size):
        #     error = errors * X[:, i]
        #     error_sum = error.sum()
        #     subtract_factor[i][0] = alpha*(1.0/m)*error_sum

        theta = theta - subtract_factor
        # print theta

        J_history[iter, 0] = costFunction(X, y, theta)

    return theta, J_history

def prediction(X, theta):
    '''Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters '''

    h = predict(X, theta)

    labels = map(lambda x: 1 if x > 0.5 else 0, h)

    return np.array(labels)


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

input = np.ones((y.size,3), dtype=np.float)
input[:, 1:3] = X[:, :2]

print input

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
iters = 5000
alpha = 0.1

theta_r, j_history = gradient_descent(input, y, theta, alpha, iters)
print theta_r

plot(np.arange(iters), j_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

#plot the training dataset
pos_indices = [i for i, x in enumerate(y) if x == 1]
neg_indices = [i for i, x in enumerate(y) if x == 0]

print theta_r[0], theta_r[1], theta_r[2]
x1 = -theta_r[0]/theta_r[1]
y1 = -theta_r[0]/theta_r[2]
print x1, y1
plt.plot(X[pos_indices, 0], X[pos_indices, 1], 'k+')
plt.plot(X[neg_indices, 0], X[neg_indices, 1], 'yo')
plt.plot([0, x1], [y1, 0], 'r')
plt.show()

#Compute accuracy on our training set
p = prediction(input, theta_r)

count = 0
for idx in range(p.size):
    if p[idx] == y[idx]:
        count +=1
print count
print 'Train Accuracy: %f' % ((count / float(y.size)) * 100.0)

#
# def map_feature(x1, x2):
#     '''
#     Maps the two input features to quadratic features.
#     Returns a new feature array with more features, comprising of
#     X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
#     Inputs X1, X2 must be the same size
#     '''
#     x1.shape = (x1.size, 1)
#     x2.shape = (x2.size, 1)
#     degree = 6
#     out = np.ones(shape=(x1[:, 0].size, 1))
#
#     m, n = out.shape
#
#     for i in range(1, degree + 1):
#         for j in range(i + 1):
#             r = (x1 ** (i - j)) * (x2 ** j)
#             out = np.append(out, r, axis=1)
#
#     return out
#
#
# #load the dataset
# data = np.loadtxt('../Data/ex2data1.txt', delimiter=',')
#
# # X = map_feature(data[:,0], data[:,1])
# input = np.ones((data.shape[0], 3), dtype=np.float)
#
#
# input[:, 1:] = data[:, 0:2]
# y = data[:, 2]
#
# #test gradient_descent
# dimension = 3
# theta = np.zeros((dimension, 1), dtype=np.float)
# iters = 10000
# alpha = 0.001
# lamda = 1
#
# theta_r, j_history = gradient_descent(input, y, theta, alpha, iters)
#
# plot(np.arange(iters), j_history)
# xlabel('Iterations')
# ylabel('Cost Function')
# show()
#
# print theta_r
#
# #Compute accuracy on our training set
# p = prediction(input, theta_r)
#
# print p
#
# count = 0
# for idx in range(p.size):
#     if p[idx] == y[idx]:
#         count +=1
# print count
# print 'Train Accuracy: %f' % ((count / float(y.size)) * 100.0)
#
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

# #Plot Boundary
# u = np.linspace(0, 100, 50)
# v = np.linspace(0, 100, 50)
# z = np.zeros(shape=(len(u), len(v)))
# for i in range(len(u)):
#     for j in range(len(v)):
#         # z[i, j] = (map_feature(np.array(u[i]), np.array(v[j])).dot(np.array(theta_r)))
#         z[i, j] = np.array([1.0, u[i], v[j]]).dot(np.array(theta_r))
# z = z.T
# plt.contour(u, v, z)
# plt.title('lambda = %f' % lamda)
# plt.xlabel('Microchip Test 1')
# plt.ylabel('Microchip Test 2')
# plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
# plt.show()