from sklearn import tree
from sklearn.linear_model import LogisticRegression
"""
A hello world example for supervised machine learning using sklearn.
6 lines of code, and that is all!
"""
if __name__ == '__main__':

    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [1, 1, 0, 0]

    # clf = tree.DecisionTreeClassifier()
    clf = LogisticRegression()

    clf.fit(features, labels)
    print clf.coef_
    print clf.intercept_

    predict = clf.predict([[160, 0], [123, 1]])
    print(predict)
