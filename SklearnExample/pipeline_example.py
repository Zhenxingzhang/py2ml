"""
================================
Practical Data Science in Python
http://radimrehurek.com/data_science_python/
================================
"""

import pandas as pd
import csv
from textblob import TextBlob
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

DATA_FILENAME = '../Data/smsspamcollection/SMSSpamCollection'


def split_into_lemmas(message):
    message = str(message).lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

if __name__ == '__main__':
    # load data set

    raw_df = pd.read_csv(DATA_FILENAME, sep='\t', quoting=csv.QUOTE_NONE,
                         names=["label", "message"])

    print(raw_df.shape)
    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(raw_df['message'], raw_df['label'], test_size=0.2)

    # # create pipeline
    pipeline = Pipeline([
        ('doc2vec', CountVectorizer(analyzer=split_into_lemmas)),
        # (),
        ('clf', LogisticRegression())
    ])

    # train pipeline
    pipeline.fit(X_train, y_train)
    # evaluate pipeline performance.
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy score: ", accuracy)
    print(cm)
