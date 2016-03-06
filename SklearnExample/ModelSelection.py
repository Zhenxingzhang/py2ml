'''
Parameter tuning for SVC
'''
import numpy as np
from sklearn import cross_validation, datasets, svm
from matplotlib import pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

# scores = [ svc.set_params(C=C).fit(X,y).score(X,y) for C in C_s]
# print C_s
# print scores
# plt.semilogx(C_s, scores)
# plt.ylim(0, 1.1)
# plt.show()
# svc.set_params(C=C_s[scores.index(max(scores))])
# print svc.fit(X,y).score(X,y)

scores = list()
scores_std = list()
for C in C_s:
    svc.C = C
    this_scores = cross_validation.cross_val_score(svc, X, y, n_jobs=-11)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

# Do the plotting
plt.figure(1, figsize=(8, 6))
plt.clf()
plt.semilogx(C_s, scores)
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()

print C_s[scores.index(max(scores))]
svc.set_params(C=C_s[scores.index(max(scores))])
print svc.fit(X,y).score(X,y)


'''
Find the optimal regularization parameter alpha in diabetes dataset
'''
from sklearn import cross_validation, datasets, linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)

scores=[]
for alpha in alphas:
    lasso.set_params(alpha=alpha)
    this_scores = cross_validation.cross_val_score(lasso, X, y, n_jobs=-1)
    scores.append(np.mean(this_scores))

best_alpha = alphas[scores.index(max(scores))]

plt.semilogx(alphas, scores)
# plt.ylim(0, 1.1)
plt.show()
print best_alpha

##############################################################################
# Bonus: how much can you trust the selection of alpha?

# To answer this question we use the LassoCV object that sets its alpha
# parameter automatically from the data by internal cross-validation (i.e. it
# performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.
lasso_cv = linear_model.LassoCV(alphas=alphas)
k_fold = cross_validation.KFold(len(X), 3)

print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")

plt.show()