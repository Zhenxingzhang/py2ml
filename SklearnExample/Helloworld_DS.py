from sklearn import tree
"""
A hello world example for supervised machine learning using sklearn.
6 lines of code, and that is all!
"""
if __name__ == '__main__':

    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [1, 1, 0, 0]
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)
    predict = clf.predict([[160, 0], [123, 1]])
    print(predict)
