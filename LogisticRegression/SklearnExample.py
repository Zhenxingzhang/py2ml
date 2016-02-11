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


def draw_decision_boundary(model, data, admitted, rejected):
    xx, yy = np.mgrid[-0.8: 1.1: .01, -0.8: 1.1: .01]
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid = map_features(grid[:, 0], grid[:, 1], order=10).T

    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(10, 8))
    # contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
    #                   vmin=0, vmax=1)
    # ax_c = f.colorbar(contour)
    # ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    ax.scatter(data[admitted, 0],
                data[admitted, 1],
                c='b', marker='+', label='admitted')
    ax.scatter(data[rejected, 0],
                data[rejected, 1],
                c='y', marker='o', label='rejected')

    plt.show()
    print f
    print "draw decision boundary"


#load the dataset
# data = np.loadtxt('../Data/ex2data1.txt', delimiter=',')
data = np.loadtxt('../Data/ex2data2.txt', delimiter=',')


# X = data[:, :2]/100.0
X = data[:, :2]

y = data[:, 2]

admitted = [i for i, x in enumerate(y) if x == 1]
rejected = [i for i, x in enumerate(y) if x == 0]

# render_exams(data, admitted, rejected)

X_ = map_features(X[:,0], X[:,1], order=10).T
# Initialize our algorithm class
alg = LogisticRegression(random_state=1, penalty='l2', C=1000000, max_iter=10000)

alg.fit(X_, y)
print 'Coefficents: ', alg.coef_
print 'Intercept" ', alg.intercept_

test_predictions = alg.predict(X_)

count = 0
for idx in range(test_predictions.size):
    if test_predictions[idx] == y[idx]:
        count +=1

print 'Train Accuracy: %f' % ((count / float(y.size)) * 100.0)


# render_exams(X, admitted, rejected)
# plt.plot(ex1, ex2, color='r', label='decision boundary');
# plt.show()

draw_decision_boundary(alg, X, admitted, rejected)