from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn import svm
iris = load_iris()
# clf = AdaBoostClassifier(n_estimators=100)
clf = linear_model.LogisticRegression()
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, iris.data, iris.target)
print scores.mean()