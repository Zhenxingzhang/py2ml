import math, numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+ math.exp(-x))

def predict(x, theta):
    return map(sigmoid, x.dot(theta))

def costFunction(X, y, theta):

    m = y.size
    predictions = predict(X, theta)

    subtraction = np.ones((y.size))

    cost_rows = -1 *(y * map(math.log, predictions)) - (subtraction-y).flatten()* map(math.log, (subtraction-prediction).flatten())
    return cost_rows.sum()/m

def gradient_descent(X, y, theta, alpha, iters):


    m = 3
    J_history = np.zeros((iters ,1))
    print "J_H shape: ", J_history.shape
    for iter in range(iters):

        predictions = predict(X, theta)

        subtract_factor = np.zeros((m,1), dtype=np.float64)

        errors = predictions - y
        print iter
        for i in range(m):
            print "   ", (errors * X[:, i]).shape
            error_sum = ( errors * X[:, i]).sum()
            subtract_factor[i][0] = alpha*(1.0/m)*error_sum

        theta = theta - subtract_factor

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

X = data[:, :2]
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

prediction = predict(input, theta)
print prediction

cost = costFunction(input, y, theta)
print cost

#test gradient_descent
theta = np.zeros((3,1))
iters = 100
alpha = 0.01

theta_r, j_history = gradient_descent(input, y , theta, alpha, iters)
print theta_r