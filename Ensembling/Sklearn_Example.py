from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn import svm
iris = load_iris()
# clf = linear_model.LogisticRegression()
# clf = svm.SVC(kernel='linear')
clf = AdaBoostClassifier(n_estimators=1000)

scores = cross_val_score(clf, iris.data, iris.target)
print scores.mean()