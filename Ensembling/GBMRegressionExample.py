import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

FIGSIZE = (11, 7)

def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)

def gen_data(n_samples=200):
    """generate training and testing data"""
    np.random.seed(15)
    X = np.random.uniform(0, 10, size=n_samples)[:, np.newaxis]
    y = ground_truth(X.ravel()) + np.random.normal(scale=2, size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = gen_data(100)

# plot ground truth
x_plot = np.linspace(0, 10, 500)

def plot_data(alpha=0.4, s=20):
    fig = plt.figure(figsize=FIGSIZE)
    gt = plt.plot(x_plot, ground_truth(x_plot), alpha=alpha, label='ground truth')

    # plot training and testing data
    plt.scatter(X_train, y_train, s=s, alpha=alpha)
    plt.scatter(X_test, y_test, s=s, alpha=alpha, color='red')
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')
    # plt.show()

annotation_kw = {'xycoords': 'data', 'textcoords': 'data',
                 'arrowprops': {'arrowstyle': '->', 'connectionstyle': 'arc'}}

# plot_data()


# from sklearn.tree import DecisionTreeRegressor
# plot_data()
# est = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)
# plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]),
#          label='RT max_depth=1', color='g', alpha=0.9, linewidth=2)

# est = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
# plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]),
#          label='RT max_depth=3', color='g', alpha=0.7, linewidth=1)
# plt.legend(loc='upper left')
# plt.show()

from itertools import islice

plot_data()
est = GradientBoostingRegressor(n_estimators=1, max_depth=1, learning_rate=1.0)
est.fit(X_train, y_train)

est2 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=1.0)
est2.fit(X_train,y_train)

print est.estimators_[0][0].feature_importances_

ax = plt.gca()
first = True

# step through prediction as we add 10 more trees.
# for pred in islice(est.staged_predict(x_plot[:, np.newaxis]), 0, est.n_estimators, 10):
#     plt.plot(x_plot, pred, color='r', alpha=0.2)
#     if first:
#         ax.annotate('High bias - low variance', xy=(x_plot[x_plot.shape[0] // 2],
#                                                     pred[x_plot.shape[0] // 2]),
#                                                     xytext=(4, 4), **annotation_kw)
#         first = False

pred = est.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred, color='r', label='GBRT max_depth=1')
ax.annotate('Low bias - high variance', xy=(x_plot[x_plot.shape[0] // 2],
                                            pred[x_plot.shape[0] // 2]),
                                            xytext=(6.25, -6), **annotation_kw)
plt.legend(loc='upper left')


pred2 = est2.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred2, color='y', label='GBRT max_depth=2')
plt.show()