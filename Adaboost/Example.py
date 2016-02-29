import adaboost
from sklearn.ensemble import AdaBoostClassifier
t = adaboost.AdaBoost()

# print detailed debugging information regarding the classifier selection
t.debug = 2

x= [[1,  2],
    [1,  4  ],
    [2.5,5.5],
    [3.5,6.5],
    [4,  5.4],
    [2,  1],
    [2,  4],
    [3.5,3.5],
    [5,  2],
    [5,  5.5]]
y=[1,1,1,1,1,-1,-1,-1,-1,-1]
# train classifier
t.train(x, y) # x is a matrix, y is a actual classifications (-1 or 1)

# classify novel set of values, the sign of the return value is predicted binary class
novel_y_prime = t.apply_to_matrix(x)
print novel_y_prime

clf = AdaBoostClassifier(n_estimators=10)
clf.fit(x, y)
print clf.feature_importances_
print clf.predict(x)