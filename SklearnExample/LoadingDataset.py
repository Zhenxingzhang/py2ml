from sklearn import datasets, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()

X_iris = iris.data
y_iris = iris.target

print iris.keys()
print iris.target_names
print iris.data.shape

# alg = LogisticRegression()

alg = MLPClassifier(activation = 'logistic', algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(4,2), random_state=1)
kfold =3
scores = cross_validation.cross_val_score(alg, X_iris, y_iris, cv=kfold, n_jobs= -1)

print scores.mean()