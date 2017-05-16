from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt


def sigmoid_function(features, b0, b1):
    return 1.0 / (1.0 + np.e ** (-b0 - b1 * features))


if __name__ == "__main__":
    print("hello world")
    X = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1]

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X, y)

    p0 = clf.intercept_[0]
    p1 = clf.coef_[0][0]
    print "parameters: {}, {}".format(p0, p1)

    x = [[0], [1], [10]]

    print "results from sklearn: {}".format(clf.predict_proba(x))

    print sigmoid_function(10, p0, p1)

    u = np.linspace(-15, 30, 100)
    ps = sigmoid_function(u, p0, p1)
    print zip(u, ps)

    plt.plot(u, ps)

    plt.plot(u, sigmoid_function(u, -4.5, 1), color ='r')
    plt.show()
