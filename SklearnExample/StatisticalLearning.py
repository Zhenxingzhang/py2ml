import numpy as np
from sklearn import datasets
from sklearn import linear_model

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
# print indices

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

'''
Curse of dimensionality: Sparsity
'''
diabetes = datasets.load_diabetes()
# print diabetes.data.shape
# print diabetes.target
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]
clf = linear_model.LinearRegression()
clf.fit(diabetes_X_train, diabetes_y_train)

prediction = clf.predict(diabetes_X_test)
# mean square error
print np.mean((prediction-diabetes_y_test)**2)
print clf.score(diabetes_X_test, diabetes_y_test)

ridge = linear_model.Ridge(alpha= 0.01)
ridge.fit(diabetes_X_train, diabetes_y_train)
prediction = ridge.predict(diabetes_X_test)
print np.mean((prediction-diabetes_y_test)**2)
print ridge.score(diabetes_X_test, diabetes_y_test)

alphas = np.logspace(-4, -1, 6)
scores = [ridge.set_params(alpha=alpha)
              .fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]

ridge.set_params(alpha= alphas[scores.index(max(scores))]).fit(diabetes_X_train, diabetes_y_train)
prediction = ridge.predict(diabetes_X_test)
print np.mean((prediction-diabetes_y_test)**2)
print ridge.score(diabetes_X_test, diabetes_y_test)


regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha )
              .fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test)
          for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)
