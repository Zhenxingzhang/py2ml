from sklearn import datasets
import matplotlib.cm as cm
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
import matplotlib.pyplot as plt

'''
Loading the dataset
'''
digits = datasets.load_digits()


'''
Dataset Exploration
'''
# print digits.DESCR
# print type(digits), digits.images.shape, digits.images[0].shape, digits.data.shape
# fig = plt.figure()
# for idx in range(20):
#     img = digits.images[idx]
#     data = digits.data[idx]
#     plt.subplot(5, 4, idx+1)
#     plt.imshow(data.reshape([8,8]), cmap= cm.Greys_r)
#     # print digits.target[idx]
# plt.show()
# print digits.data.max(), digits.data.min()

'''
Training svm one-vs-all model for the 10 classes
'''
#
# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(digits.data[:-10], digits.target[:-10])
#
# print digits.target[-10:]
# print clf.predict(digits.data[-10:])
# s = joblib.dump(clf, "svm.pkl")

'''
Loading svm one-vs-all model for the 10 classes
'''
# clf = joblib.load("svm.pkl")
# print digits.target[-10:]
# print clf.predict(digits.data[-10:])

'''
Sklearn library can handle different target values, (int or str)
'''
iris= datasets.load_iris()

clf = svm.SVC()
clf.set_params(kernel = 'linear').fit(iris.data[:-10], iris.target[:-10])
print clf.predict(iris.data[-10:])

print type(iris.target_names)
clf.set_params(kernel='rbf').fit(iris.data[:-10], iris.target_names[iris.target][:-10])
print clf.predict(iris.data[-10:])