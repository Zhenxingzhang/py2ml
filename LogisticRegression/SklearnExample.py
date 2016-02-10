import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def render_exams(data, admitted, rejected):
    plt.figure(figsize=(6, 6))

    plt.scatter(data[admitted, 0],
                data[admitted, 1],
                c='b', marker='+', label='admitted')
    plt.scatter(data[rejected, 0],
                data[rejected, 1],
                c='y', marker='o', label='rejected')
    plt.xlabel('Exam 1 score');
    plt.ylabel('Exam 2 score');
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend();


def map_features(f1, f2, order=1):
    '''map the f1 and f2 to its higher order polynomial'''
    assert order >= 1
    def iter():
        for i in range(1, order + 1):
            for j in range(i + 1):
                yield np.power(f1, i - j) * np.power(f2, j)
    return np.vstack(iter())

#load the dataset
data = np.loadtxt('../Data/ex2data1.txt', delimiter=',')

X = data[:, :2]/100.0
y = data[:, 2]

admitted = [i for i, x in enumerate(y) if x == 1]
rejected = [i for i, x in enumerate(y) if x == 0]

render_exams(data, admitted, rejected)

# X =map_features(X[:,0], X[:,1], order=2).T
# Initialize our algorithm class
alg = LogisticRegression(random_state=1, penalty='l1')

alg.fit(X, y)
print 'Coefficents: ', alg.coef_
print 'Intercept" ', alg.intercept_

test_predictions = alg.predict(X)

count = 0
for idx in range(test_predictions.size):
    if test_predictions[idx] == y[idx]:
        count +=1
print count
print 'Train Accuracy: %f' % ((count / float(y.size)) * 100.0)

coef = alg.coef_
intercept = alg.intercept_

# see the coutour approach for a more general solution
ex1 = np.linspace(-1, 1.5, 10)
ex2 = -(coef[:, 0] * ex1 + intercept[0]) / coef[:,1]

render_exams(X, admitted, rejected)
plt.plot(ex1, ex2, color='r', label='decision boundary');
plt.show()