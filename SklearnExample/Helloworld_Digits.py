from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import logging
import matplotlib.pyplot as plt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    digits = datasets.load_digits()
    logger.info("Dataset structure: {}".format(digits.keys()))

    logger.info("Dataset: {}".format(digits.data.shape))

    raw_data = digits.data
    labels = digits.target

    X_train, X_test, y_train, y_test = train_test_split(raw_data, labels, test_size=0.2, random_state=0)

    clf = SVC()
    tuned_parameters1 = {'kernel': ['linear'], 'C': [0.1, 1, 10]}
    tuned_parameters2 = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [1e-3, 1e-4]}
    tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [1e-3, 1e-4]},
                        {'kernel': ['linear'], 'C': [0.1, 1, 10]}]

    y_preds_pd = pd.DataFrame()
    for parameters in [tuned_parameters1, tuned_parameters2]:
        gd_cv = GridSearchCV(clf, parameters, cv=5, refit=True, n_jobs=-1, scoring='accuracy')
        gd_cv.fit(X_train, y_train)

        logger.info("Best accuracy score: {}".format(gd_cv.best_score_))
        logger.info('Best parameters: {}'.format(gd_cv.best_params_))

        best_clf = gd_cv.best_estimator_

        y_pred = best_clf.predict(X_test)
        y_preds_pd[parameters['kernel'][0]] = y_pred

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        logger.info("Accuracy on testing dataset: {}".format(accuracy))

        logger.info("Confusion matrix: \n{}".format(confusion_matrix(y_test, y_pred)))

    pred_diff = y_preds_pd[y_preds_pd['linear'] != y_preds_pd['rbf']]

    logger.info(pred_diff)

    for idx in pred_diff.index:
        img = X_test[idx, :].reshape([8, 8])
        # logger.info(img)
        plt.figure(figsize=(2, 2))
        plt.imshow(img, interpolation='nearest', cmap=plt.cm.seismic)
        plt.axis('off')
        plt.gray()
        plt.show()

