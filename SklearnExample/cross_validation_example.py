import numpy as np
from sklearn import cross_validation, datasets, svm
import matplotlib.pyplot as plt
# % matplotlib inline

digits = datasets.load_digits()
print(digits.keys())

print(digits.data.shape)

digits = datasets.load_digits()
X = digits.data
y = digits.target

clf = svm.SVC(kernel='linear')

C_s = np.logspace(-10, 2, 10)
# C_s = np.arange(0.001, 1.0, 0.1)

train_score = []
cv_scores = []

for c_s in C_s:
    clf.set_params(C=c_s)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    cv_scores.append(scores.mean())

    clf.fit(X, y)
    score = clf.score(X, y)
    train_score.append(score)

print(C_s)
print(train_score)
print(cv_scores)

plt.plot(C_s, train_score, 'r--', label="train_score")
plt.plot(C_s, cv_scores, 'b-', label="cv_score")
plt.xscale("log")
# plt.xticks(np.arange(10),C_s)
# fig.set_xticklabels(C_s, rotation='vertical')
plt.show()

# print("accuracy score : {}").format(score)
