import numpy as np
from sklearn import datasets

'''
Supervised Learning
'''
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print np.unique(iris_y)

'''
split the data into train and test set with a random permutation
'''
indices = np.random.permutation(len(iris_y))
print indices

iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
print clf
clf.fit(iris_X_train, iris_y_train)

predict_labels = clf.predict(iris_X_test)

print iris_y_test
print predict_labels
