#!/usr/bin/env python3
import os
import json
import argparse
import sys
import numpy as np
import pandas as pd

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
    parser.add_argument('--train', help='file name for training data')
    parser.add_argument('--test', help='file name for test data')
    args = parser.parse_args()
    if args.train:
        train_file = filter_corpus(args.train)
        X_train, y_train = prepare_data(train_file)
    if args.test:
        test_file = filter_corpus(args.test)
        X_test, y_test = prepare_data(test_file)
    else:
        print("No data source found! Specify which data the script"
              " should use.")
        parser.print_help()
        exit()  
    labels = np.unique(y_test)
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
    features = [row["headline"] for row in X_train]
    pipeline.fit(features, y_train)

    print("Applying algorithm to test data...")
    samples = [row["headline"] for row in X_test]
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
    
POLITICAL_ORIENTATIONS = {
    'Sydney Morning Herald (Australia)': 'left',
    'The Age (Melbourne, Australia)': 'left',
    'The Hindu': 'left',
    'Mail & Guardian': 'left',
    'The New York Times': 'left',
    'The Washington Post': 'left',
    'The Australian': 'right',
    'The Times of India (TOI)': 'right',
    'The Times (South Africa)': 'right'
}
 

def get_political_orientation(newspaper):
    """
    Wrapper function for retrieving the political
    orienation of a newspaper.
    :param newspaper:
    :return:
    """
    return POLITICAL_ORIENTATIONS[newspaper]
       
def filter_corpus(f):
    """
    Gathers information from all the JSON files
    and only returns the relevant keys/fields for our
    project.
    :return:
    """
    result = []
    with open(f, "r") as infile:
        data = json.load(infile)
        common_info = {
                "cop_edition": data["cop_edition"],
                "collection_start": data["collection_start"],
                "collection_end": data["collection_end"],
            }

        for article in data['articles']:
            new_article = {
                    "newspaper": article["newspaper"],
                    "political_orientation": get_political_orientation(
                        article["newspaper"]),
                    "headline": article["headline"],
                    "date": article["date"],
                    "body": article["body"]
                }
            row = {**common_info, **new_article}
            result.append(row)
        return(json.dumps(result, indent=2))

def prepare_data(data):
    """
    Prepares classification data
    by applying filters and splitting.
    :param data:
    :return:
    """
    data = json.loads(data)
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
