#!/usr/bin/env python3
import json
import os
import argparse
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score, accuracy_score, \
    precision_recall_fscore_support, confusion_matrix, precision_score, \
    recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

import pandas as pd
import numpy as np

AMERICAN_NEWSPAPERS = {'The New York Times'} # ,'The Washington Post'}


def main():
    """
    Performs the classification of political orientations
    of corpora from the COP collection.
    """
    parser = argparse.ArgumentParser(
        description='Parameters for the COP Political Orientation'
                    ' classifier')
    parser.add_argument('--train', help='file name for training data')
    parser.add_argument('--test', help='file name for test data')
    parser.add_argument('-f', '--full', help="file name for complete"
                                             " data set")
    args = parser.parse_args()

    # TODO add handling for specific test/training data
    if args.full:
        with open(args.full, 'r') as F:
            data = json.load(F)
    else:
        print("No data source found! Specify which data the script"
              " should use.")
        parser.print_help()
        exit()

    X, y = [], []
    for row in data:
        X.append({key: value for key, value in row.items()
                  if key != "political_orientation"})
        y.append(row["political_orientation"])

    for row in X:
        if row["newspaper"] in AMERICAN_NEWSPAPERS:
            index = X.index(row)
            del X[index]
            del y[index]

    labels = np.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        test_size=0.27)

    # set up parameter grid
    parameters = {
        'vec__strip_accents': [None],
        'vec__lowercase': [True, False],
        'vec__analyzer': ['word'],
        'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vec__norm': ['l2', 'l1'],
        'clf__alpha': [0, 0.001, 0.1, 0.01, 0.25],
    }

    # set up classifier pipeline
    pipeline = Pipeline([
        ('vec', TfidfVectorizer()),
        ('clf', BernoulliNB())
    ])

    score_type = "macro"  # niet 'weighted' n.a.v. bericht Jantina
    # set up scorer, so we can compare both F1 and accuracy scores
    scoring = {'F1': make_scorer(f1_score, average=score_type),
               'Accuracy': make_scorer(accuracy_score)}

    classifier = GridSearchCV(pipeline, parameters,
                              scoring=scoring,
                              n_jobs=-1, cv=3,
                              return_train_score=False,
                              refit='F1',
                              verbose=10)

    features = [row['headline'] for row in X_train]
    classifier.fit(features, y_train)

    samples = [row["headline"] for row in X_test]
    y_guess = classifier.predict(samples)

    # Prints the scores.
    print("Overall scores:")
    print("Accuracy", accuracy_score(y_true=y_test, y_pred=y_guess))
    print("Precision", precision_score(y_true=y_test, y_pred=y_guess,
                                       average=score_type))
    print("Recall", recall_score(y_true=y_test, y_pred=y_guess,
                                 average=score_type))
    print("F1-score", f1_score(y_true=y_test, y_pred=y_guess,
                               average=score_type))
    print()

    scores = precision_recall_fscore_support(y_test, y_guess, labels=labels)
    print(pd.DataFrame(scores, columns=labels,
                       index=["Precision", "Recall", "F-score",
                              "Support"]).drop(["Support"]), '\n')

    # Print confusion matrix.
    matrix = confusion_matrix(y_test, y_guess, labels=labels)
    print(pd.DataFrame(matrix, index=labels, columns=labels), '\n')

    print("Best parameters:")
    print(classifier.best_params_)

    results = pd.DataFrame(classifier.cv_results_)
    results = results.sort_values('mean_test_F1', ascending=False)
    if not os.path.isdir('results'):
        os.mkdir('results')
    results.to_excel("results/nb_bernoulli.xlsx", engine='openpyxl')
    results.to_html("results/nb_bernoulli.html")

    joblib.dump(classifier, "models/nb_bernoulli.joblib")


if __name__ == '__main__':
    main()
