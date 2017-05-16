import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class LinearRegression:
    weights =[]
    iteration = 10

    def __loss(self, y, y_hat):
        return np.sum(np.array(y-y_hat)**2)

    def __preprocess(self, data):
        N = len(data)
        bias = np.ones((N, 1))
        return np.c_[bias, np.array(data)]

    def train(self, X, y, a=0.1):
        m = len(X[0])
        self.weights = np.zeros(m + 1)
        loss = np.zeros(self.iteration)

        train_X= self.__preprocess(X)

        y_init = self.predict(X)
        loss[0] = self.__loss(y, y_init)

        for iter in range(self.iteration):
            y_hat = self.predict(X)

            self.weights[0] += a * np.sum(np.array(y-y_hat))
            self.weights[1] += a * np.sum(np.array(y-y_hat) * train_X[:, 1])

            loss[iter] = self.__loss(y, self.predict(X))
        return loss

    def predict(self, input):
        return np.array(self.__preprocess(input)).dot(self.weights)

if __name__ == "__main__":
    lr = LinearRegression()
    X = [[1], [2]]
    y = [1.1, 1.9]

    print lr.train(X, y)

    plt.scatter([1,2], lr.predict(X))
    plt.show()
