from __future__ import print_function
from numpy import zeros, ones
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.linear_model as sk

def compute_cost(feature_x, predict_y, params):
    """
    :param feature_x: the input data vector [m,d]
    :param predict_y: the groundtruth output[m,1]
    :param params: the weights [d+1, 1]
    :return: the squared error
    """
    length = predict_y.size

    predictions = y_function(feature_x, params).flatten()
    sqr_errors = (predictions - predict_y) ** 2

    loss = (1.0/(2 * length)) * sqr_errors.sum()
    return loss


def gradient_descent(feature_x, predict_y, params, alpha_, num_iters):
    """

    :param feature_x:
    :param predict_y:
    :param params:
    :param alpha_:
    :param num_iters:
    :return:
    """

    loss_history = zeros(shape=(num_iters,))

    m = predict_y.size
    for iter in range(num_iters):
        predictions = y_function(feature_x, params[:, iter]).flatten()

        errors_x1 = (predictions - predict_y) * feature_x[:, 0]
        errors_x2 = (predictions - predict_y) * feature_x[:, 1]

        params[0][iter + 1] = params[0][iter] - alpha_ * (1.0 / m) * errors_x1.sum()
        params[1][iter + 1] = params[1][iter] - alpha_ * (1.0 / m) * errors_x2.sum()

        loss_history[iter] = compute_cost(feature_x, predict_y, params[:, iter])

    return params, loss_history


def y_function(feature, params):
    return feature.dot(params)


def sigmoid_function(features):
    return 1.0 / (1.0 + np.e ** (-1.0 * features))


def log_function(predicts):
    return np.log(predicts)


def logit_function(input):
    return np.log(input) - np.log(1-input)

def sigmoid(x):
    return 1.0 /(1.0 + np.exp(-1.0 * x))


def predict(features, params):
    # return np.exp(features.dot(params))
    return 800 * sigmoid(features.dot(params)).flatten()

if __name__ == "__main__":
    temp = [11.9, 14.2, 15.2, 16.4, 17.2, 18.1, 18.5, 19.4, 22.1, 22.6, 23.4, 25.1]
    units = [185, 215, 332, 325, 408, 421, 406, 412, 522, 445, 544, 614]
    units = np.asarray(units)

    # 0. linear regression

    # 1. log-normal transformation

    # y_units = log_function(np.asarray(units))

    # 3. Binomial regression
    units_probability = units/800.0
    print(units_probability)

    y_units = logit_function(units_probability)
    print(y_units)

    plt.scatter(temp, y_units)
    plt.show()

    iterations = 20000
    alpha = 0.005

    theta = zeros(shape=(2, iterations+1))

    # Add a column of ones to X (interception data)
    length = len(temp)

    X = ones(shape=(length, 2))
    X[:, 1] = np.transpose(np.asarray(temp))

    theta, loss = gradient_descent(X, y_units, theta, alpha, iterations)
    print("final loss: {}".format(loss[-1]))
    plt.plot(range(iterations), loss)
    plt.show()

    # alg = sk.LinearRegression()
    # alg.fit(X, y)
    # result = alg.predict(X)
    # print(alg.coef_)
    # print(alg.intercept_)

    train_params = theta[:, iterations]
    # train_params = [4.40209035097, 0.08260077]
    print(theta)
    print(train_params)
    result = predict(X, train_params)

    plt.scatter(X[:, 1], units)
    plt.plot(X[:, 1], result)
    plt.show()

    # TODO sklearn lib works fine, not my code, need inspection! solution: increasing the train iteration

    points_x = np.arange(-5, 50, 1)
    points = np.ones(shape=(points_x.size, 2))
    points[:, 1] = points_x
    # print(points.reshape([n,1]))
    values = predict(points, train_params)

    plt.plot(points_x, values)
    plt.show()
