from sklearn import datasets
import matplotlib.cm as cm
import numpy as np

import matplotlib.pyplot as plt

digits = datasets.load_digits()

print type(digits), digits.images.shape, digits.images[0].shape, digits.data.shape

img = digits.images[1]

data = digits.data[1]

data = np.reshape(data, (8,8))

print data.shape

fig = plt.figure()
for idx in range(20):
    img = digits.images[idx]
    plt.subplot(5, 4, idx+1)
    plt.imshow(img, cmap = cm.Greys_r)
    # print digits.target[idx]

plt.show()

# fig = plt.figure()
#
# plt.subplot(211)
# plt.imshow(img, cmap = cm.Greys_r)
#
# plt.subplot(212)
# plt.imshow(data, cmap = cm.Greys_r)
# plt.show()
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-10], digits.target[:-10])
print digits.target[-10:]
print clf.predict(digits.data[-10:])