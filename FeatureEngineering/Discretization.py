import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
#Generate synthetic data
from sklearn.linear_model import LogisticRegression as LR
# mu, sigma = 2, 2
# s = np.random.normal(mu, sigma, 1000)
#
# count, bins, ignored = plt.hist(s, 30, normed=False)
# train_data = np.array(zip(range(1, 31, 1), count))
# print count
# print train_data.shape

def draw_decision_boundary(model, data =[], label=[]):
    xx, yy = np.mgrid[0.0: 3: .1, 0: 1.5: .1]
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(10, 8))
    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    plt.scatter(data[:6, 0], data[:6, 1], c='red', marker='+')
    plt.scatter(data[6:, 0], data[6:, 1], c='black', marker='o')
    plt.show()
    print f
    print "draw decision boundary"


train_data = [[0.12, 0.74], [0.3, 1.0],[1.25,1.0], [1.74, 0.4], [2.65, 0.8],[2.95,2.3],
              [0.5, 0.48], [0.8,1.0], [1.2, 0.6], [1.80, 0.08], [2.49,0.10],[2.90, 1.05]]
train_data=np.array(train_data)
y_label = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
# y_label = [1]*30
#
# y_label =
#
clg = LR()
clg.fit(train_data, y_label)

draw_decision_boundary(clg, train_data)
