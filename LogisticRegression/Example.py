import numpy

def sigmoid(X):
    return 1 / (1 + numpy.exp(- X))

def cost(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta)) # predicted probability of label 1
    log_l = (-y)*numpy.log(p_1) - (1-y)*numpy.log(1-p_1) # log-likelihood vector

    return log_l.mean()

def grad(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta))
    error = p_1 - y # difference between label and prediction
    grad = numpy.dot(error, X_1) / y.size # gradient vector

    return grad

def predict(X, theta,):
    '''Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters '''
    m, n = X.shape
    p = numpy.zeros(shape=(m, 1))

    h = sigmoid(X.dot(theta))

    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0

    return p

import scipy.optimize as opt
import matplotlib.pyplot as plt

#load the dataset
data = numpy.loadtxt('../Data/ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
# prefix an extra column of ones to the feature matrix (for intercept term)
theta = 0.1* numpy.random.randn(3)
X_1 = numpy.append( numpy.ones((X.shape[0], 1)), X, axis=1)

theta_1 = opt.fmin_bfgs(cost, theta, fprime=grad, args=(X_1, y))

#Compute accuracy on our training set
print theta_1

p = predict(X_1, theta_1)

print p

count = 0
for idx in range(p.size):
    if p[idx] == y[idx]:
        count +=1
print count
print 'Train Accuracy: %f' % ((count / float(y.size)) * 100.0)

#plot the training dataset
pos_indices = [i for i, x in enumerate(y) if x == 1]
neg_indices = [i for i, x in enumerate(y) if x == 0]

x1 = theta_1[0]/-theta_1[1]
y1 = theta_1[0]/-theta_1[2]
print x1, y1
plt.plot(X[pos_indices, 0], X[pos_indices, 1], 'k+')
plt.plot(X[neg_indices, 0], X[neg_indices, 1], 'yo')
plt.plot([0, x1], [y1, 0], 'r')
# plt.axis([0.3, 1, 0.3, 1])
plt.show()