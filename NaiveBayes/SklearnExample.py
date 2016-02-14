from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

print iris.data.shape
print iris.target

gnb = GaussianNB()

y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

print accuracy_score(iris.target, y_pred)