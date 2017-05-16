import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from tqdm import tqdm

FIGSIZE = (11, 7)

x_plot = np.linspace(0, 10, 500)

def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)

def gen_data(n_samples=200):
    """generate training and testing data"""
    np.random.seed(15)
    x_ = np.random.uniform(0, 10, size=n_samples)[:, np.newaxis]

    y_ = ground_truth(x_.ravel()) + np.random.normal(scale=2, size=n_samples)
    # train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    # x_train_, x_test_, y_train_, y_test_ = train_test_split(x_, y_, test_size=0.2, random_state=3)
    # return x_train_, x_test_, y_train_, y_test_
    return train_test_split(x_, y_, test_size=0.2, random_state=3)

def plot_data(x_, y_, color = "red", alpha=0.4, s=20):

    plt.figure(figsize=FIGSIZE)
    plt.plot(x_plot, ground_truth(x_plot), alpha=alpha, label='ground truth')

    # plot training and testing data
    plt.scatter(x_, y_, alpha=alpha, color=color)
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')


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


# 1. Loading dataset
# 2. Extracting features
# 3. Training a classifier
#    Parameter tuning using grid search
# 4. Evaluation of the performance on the test set
def grid_search(alg, param_grid):
    X_train, X_test, y_train, y_test = gen_data(200)

    gs_cv = GridSearchCV(alg, param_grid, cv=5, refit=True).fit(X_train, y_train)

    print gs_cv.best_estimator_
    print gs_cv.best_score_
    # print gs_cv.scorer_
    # print gs_cv.score(X_test, y_test)

    preds = gs_cv.best_estimator_.predict(X_test)
    print mean_absolute_error(y_test, preds)
    print r2_score(y_test, preds)


def experiment():
    param_grids = [
        {
            "learning_rate": [0.1, 0.05, 0.02, 0.01],
            "min_samples_leaf": [3, 5],
            "subsample": [0.5, 0.8]
        },
        {
            "max_depth": [2, 3, 4, 5, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3]
        },
        {
            "max_depth": [3, 4, 5, 6],
            "n_estimators": [10, 50],
            "min_samples_leaf": [1, 3, 5]
        },
        dict(feature__degree=[6, 7, 8])
    ]

    algs = [
        GradientBoostingRegressor(n_estimators=1000, max_depth=1, loss='lad'),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        Pipeline([('feature', PolynomialFeatures()), ('clg', Ridge())])
    ]

    for alg, param_grid in zip(algs, param_grids):
        # print alg, param_grid
        grid_search(alg, param_grid)


def explain():
    X_train, X_test, y_train, y_test = gen_data(200)
    # print X_train.shape
    #  plot_data(X_train, y_train)
    # plt.show()

    dt_train_loss = []
    dt_test_loss =[]

    gbt_train_loss = []
    gbt_test_loss =[]

    iteration = 2

    for idx in tqdm(range(1, iteration)):
        dt_alg = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)

        dt_train_loss.append(mean_squared_error(y_train, dt_alg.predict(X_train)))
        dt_test_loss.append(mean_squared_error(y_test, dt_alg.predict(X_test)))

        gbt_alg = GradientBoostingRegressor(n_estimators=idx, max_depth=1, learning_rate=0.05, subsample=0.8)
        gbt_alg.fit(X_train, y_train)

        gbt_train_loss.append(mean_squared_error(y_train, gbt_alg.predict(X_train)))
        gbt_test_loss.append(mean_squared_error(y_test, gbt_alg.predict(X_test)))

    fig, ax = plt.subplots()

    plt.plot(range(1, iteration), dt_train_loss, label="DT_train")
    plt.plot(range(1, iteration), dt_test_loss, label="DT_test")

    plt.plot(range(1, iteration), gbt_train_loss, label="GBT_train")
    plt.plot(range(1, iteration), gbt_test_loss, label="GBT_test")

    legend = ax.legend(loc='upper center', shadow=True)

    plt.show()

if __name__ == "__main__":
    # experiment()
    explain()


def main():
    X_train, X_test, y_train, y_test = gen_data(100)


    plot_data()
    est = GradientBoostingRegressor(n_estimators=1, max_depth=1, learning_rate=1.0)
    est.fit(X_train, y_train)

    est2 = GradientBoostingRegressor(n_estimators=8000, max_depth=1, subsample=0.5, learning_rate=0.01)
    est2.fit(X_train, y_train)

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
    plt.plot(x_plot, pred2, color='y', label='GBRT max_depth=1, n_estimator=2')
    plt.show()