__author__ = 'zhenxing'

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier

# X=[ [1,0,0], [1,0,1] ,[1,1,0] , [1,1,1]]
# y=[1,0,0,1]
#
# clf = MLPClassifier(activation = 'logistic', algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=1)
# clf.fit(X, y)
#
# print [coef.shape for coef in clf.coefs_]
#
# print clf.predict([[0 , 1 ], [0 , 0]])

from sklearn import datasets, svm, neighbors, cross_validation
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print type(X_digits), X_digits.shape

n_neighbors = 5
alg = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")

# alg = LogisticRegression(C=1e5)

# alg = svm.SVC(C=1, kernel="linear")

# alg = MLPClassifier(activation = 'logistic', algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(64, 64), random_state=1)

# alg.fit(X_digits, y_digits)

# print [coef.shape for coef in alg.coefs_]

kfold = 3

scores = cross_validation.cross_val_score(alg, X_digits, y_digits, cv=kfold, n_jobs= -1)

print scores.mean()

# scores = []
# hidden_neurals = [idx for idx in range(1, 80, 5)]
# for hidden_neural in hidden_neurals:
#     alg = MLPClassifier(activation = 'logistic', algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(hidden_neural), random_state=1)
#     cross_scores = cross_validation.cross_val_score(alg, X_digits, y_digits, cv=kfold, n_jobs= -1)
#     scores.append(cross_scores.mean())
# print scores
#
# plt.scatter(hidden_neurals, scores)
# plt.xlabel("Number of hidden neurals")
# plt.ylabel("Cross validation score")
# plt.show()