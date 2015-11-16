import math, numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

def sigmoid(x):
    return 1.0 /(1.0 + math.exp(-1.0 * x))

def predict(x, theta):
    return map(sigmoid, x.dot(theta))

def costFunction(X, y, theta, lamda):

    m = y.size
    predictions = predict(X, theta)

    subtraction = np.ones((y.size), dtype=np.float)

    cost_rows = -1.0 *(y * map(math.log, predictions)) - (1-y)* map(math.log, (1-predictions))

    regularized = ( (theta[1:]**2).sum() * lamda) / (2.0*m)
    print "regularization: ", regularized
    return cost_rows.sum()/m + regularized
    # return cost_rows.sum()/m

def gradient_descent(X, y, theta, alpha, lamda, iters):

    m = y.size
    J_history = np.zeros((iters ,1))

    for iter in range(iters):

        predictions = predict(X, theta)

        errors = predictions - y

        errors_sum_1 =alpha*(1.0 / m) * (errors*X[:, 0]).sum()

        theta[0] = theta[0] - errors_sum_1

        subtract_factor = np.zeros((theta.size-1,1), dtype=np.float)

        for j in range(1, theta.size):
            error = errors * X[:, j]
            error_sum = error.sum()
            subtract_factor[j-1][0] = alpha*(1.0/m)*(error_sum + lamda * theta[j])
            # subtract_factor[j-1][0] = alpha*(1.0/m)*(error_sum)

        theta[1:] = theta[1:] - subtract_factor

        J_history[iter, 0] = costFunction(X, y, theta, lamda)

    return theta, J_history

def prediction( X, theta):
    '''Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters '''
    m, n = X.shape
    p =np.zeros(shape=(m, 1))

    h = predict(X, theta)

    for it in range(m):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0

    return p

def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out


# #test sigmoid function
# input = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
# result = map(sigmoid, input)
# plt.plot(input, result)
# plt.ylabel('Sigmoid')
# plt.show()


#load the dataset
data = np.loadtxt('../Data/ex2data2.txt', delimiter=',')

X = data[:, :2]
y = data[:, 2]
print y.shape
#plot the training dataset
pos_indices = [i for i, x in enumerate(y) if x == 1]
neg_indices = [i for i, x in enumerate(y) if x == 0]

plt.plot(X[pos_indices, 0], X[pos_indices, 1], 'k+')
plt.plot(X[neg_indices, 0], X[neg_indices, 1], 'yo')
plt.show()

input = map_feature(X[:, 0], X[:,1])


theta = np.zeros((28,1), dtype=np.float)

lamda = 0

print '0.693 = ', costFunction(input, y, theta, lamda)

iters = 1000
alpha = 1
lamda = 0
#
theta_r, j_history = gradient_descent(input, y, theta, alpha, lamda, iters)

plot(np.arange(iters), j_history)
xlabel('Iterations')
ylabel('Cost Function')
show()



#Compute accuracy on our training set
p = prediction(input, theta_r)

count = 0
for idx in range(p.size):
    if p[idx] == y[idx]:
        count +=1
print count
print 'Train Accuracy: %f' % ((count / float(y.size)) * 100.0)

#Plot Boundary
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = (map_feature(np.array(u[i]), np.array(v[j])).dot(np.array(theta_r)))

z = z.T
plt.contour(u, v, z)
plt.title('lambda = %f' % lamda)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
plt.show()