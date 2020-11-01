#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

EXCLUDED_NEWSPAPERS = {'The New York Times'} # ,'The Washington Post'}
SCORE_TYPE = "macro"


def main():
    """
    Classifies a given corpus by political orientation.
    Part of the COPPOC project.

    The parameter values of the vectorizer and the classifier
    have been validated in a GridSearch setup by Leon
    on the 28th of October.
    :return:
    """
    parser = argparse.ArgumentParser(
        description='Parameters for the COP Political Orientation'
                    ' classifier')
    # parser.add_argument('--train', help='file name for training data')
    # parser.add_argument('--test', help='file name for test data')
    parser.add_argument('-f', '--full', help="file name for complete"
                                             " data set")
    args = parser.parse_args()

    # TODO add handling for specific test/training data
    if args.full:
        print("Loading data...")
        with open(args.full, 'r') as F:
            data = json.load(F)
    else:
        print("No data source found! Specify which data the script"
              " should use.")
        parser.print_help()
        exit()

    X, y = prepare_data(data)
    labels = np.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        test_size=0.27)
    print("Data has been split!")

    vectorizer = TfidfVectorizer(analyzer='word', lowercase=False,
                                 ngram_range=(1, 2), norm='l2',
                                 strip_accents=None)
    classifier = LinearSVC(C=0.7, dual=True, fit_intercept=True, loss='hinge',
                           multi_class='crammer_singer', penalty='l1')

    pipeline = Pipeline([
        ('vec', vectorizer),
        ('clf', classifier)
    ])

    print("Transforming data...")
    features = [row["body"] for row in X_train]
    pipeline.fit(features, y_train)

    print("Applying algorithm to test data...")
    samples = [row["body"] for row in X_test]
    y_guess = pipeline.predict(samples)
    print("Classification completed!\n")

    # Prints the scores.
    print(f"Overall scores ({SCORE_TYPE}):")
    print("Accuracy", accuracy_score(y_true=y_test, y_pred=y_guess))
    print("Precision", precision_score(y_true=y_test, y_pred=y_guess,
                                       average=SCORE_TYPE))
    print("Recall", recall_score(y_true=y_test, y_pred=y_guess,
                                 average=SCORE_TYPE))
    print("F1-score", f1_score(y_true=y_test, y_pred=y_guess,
                               average=SCORE_TYPE))
    print()

    scores = precision_recall_fscore_support(y_test, y_guess, labels=labels)
    print(pd.DataFrame(scores, columns=labels,
                       index=["Precision", "Recall", "F-score",
                              "Support"]).drop(["Support"]), '\n')

    # Print confusion matrix.
    matrix = confusion_matrix(y_test, y_guess, labels=labels)
    print(pd.DataFrame(matrix, index=labels, columns=labels), '\n')

    dump(pipeline, "classification_pipeline.joblib")

    results = pd.DataFrame(scores)
    # results = results.sort_values('mean_test_F1', ascending=False)
    results.to_excel("main_scores.xlsx", engine='openpyxl')
    results.to_html("main_scores.html")


def prepare_data(data):
    """
    Prepares classification data
    by applying filters and splitting.
    :param data:
    :return:
    """
    X, y = [], []
    for row in data:
        X.append({key: value for key, value in row.items()
                  if key != "political_orientation"})
        y.append(row["political_orientation"])

    for row in X:
        if row["newspaper"] in EXCLUDED_NEWSPAPERS:
            index = X.index(row)
            del X[index]
            del y[index]

    return X, y


if __name__ == '__main__':
    main()
