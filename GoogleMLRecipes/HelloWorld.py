from sklearn import tree
from matplotlib import pyplot as plt

import numpy as np

# Collect training data
data = [[150, 0], [170, 0], [140, 1], [130, 1]]
label = [0, 0, 1, 1]

test = [[160, 0], [170, 1]]
# Train Classifier
clf = tree.DecisionTreeClassifier()
clf.fit(data, label)

# Make predictions
y_test = clf.predict(test)

print(y_test)


# create a mesh to plot in
# x_min, x_max = 120, 180
# y_min, y_max = 0, 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 10),
#                      np.arange(y_min, y_max, 0.1))
#
# print(zip(xx, yy))
# points = {(x, y) for x in np.arange(120, 180, 10.0) for y in np.arange(0, 1, 0.1)}
# print(points)