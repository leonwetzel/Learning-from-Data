import sys
import time
import os
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer,\
    TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, \
    precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)


def read_corpus(corpus_file):
    """
    Reads information from a given file, using sentiment if indicated.
    We assume a provided file consists of 1) a genre, 2) a sentiment
    type and 3) a review.
    :param corpus_file:
    :param use_sentiment:
    :return:
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            # 6-class problem:
            # books, camera, dvd, health, music, software
            labels.append(tokens[0])

    return documents, labels


def identity(x):
    """
    A dummy function that just returns its input.
    :param x:
    :return:
    """
    return x


def main():
    """
    This script reads text from a given
    file and predicts either the sentiment type
    or the genre, based on a provided flag setting '-s'.
    """
    try:
        # check if training file exists
        if sys.argv[1] and os.path.isfile(sys.argv[1]):
            Xtrain, Ytrain = read_corpus(sys.argv[1])
        # check if test file exists
        if sys.argv[2] and os.path.isfile(sys.argv[2]):
            Xtest, Ytest = read_corpus(sys.argv[2])
    except IndexError:
        # catch cases where arguments are given
        raise FileNotFoundError("Please enter both training"
                                " and test file names.")

    # Load and split dataset
    X, Y = read_corpus('trainset.txt')
    split_point = int(0.75 * len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]
    labels = np.unique(Ytest)

    # Let's use the TF-IDF vectorizer
    tfidf = True

    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity,
                              tokenizer=identity)
    else:
        vec = CountVectorizer(preprocessor=identity,
                              tokenizer=identity)

    # TODO: GridSearchCV implementeren

    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([
        ('vec', vec),
        ('cls', MultinomialNB())
    ])

    parameters = {
        'vec__strip_accents': ['ascii', 'unicode', None],
        'vec__lowercase': [False, True],
        'vec__analyzer': ['word', 'char', 'char_wb'],
        'vec__stop_words': ['english', None],
        'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vec__norm': ['l1', 'l2'],
        'clf__alpha': [1.0, 0, 0.5, 0.75],
    }

    pipeline = Pipeline([
        ('vec', vec),
        ('clf', MultinomialNB())
    ])

    # set up scorer, so we can compare both F1 and accuracy scores
    scoring = {'F1': make_scorer(f1_score, average='weighted'),
               'Accuracy': make_scorer(accuracy_score)}

    classifier = GridSearchCV(pipeline, parameters,
                              scoring=scoring,
                              n_jobs=-1, cv=5,
                              return_train_score=False,
                              refit='F1',
                              verbose=100)

    t0 = time.time()

    # Trains the classifier, by feeding documents (X)
    # and labels (y).
    classifier.fit(Xtrain, Ytrain)

    current_time = time.time() - t0
    print(current_time)

    # Classifier makes a prediction, based on
    # a test sample of documents.
    Yguess = classifier.predict(Xtest)

    # Prints the scores.
    print("Overall scores")
    print("Accuracy", accuracy_score(y_true=Ytest, y_pred=Yguess))
    print("Precision", precision_score(y_true=Ytest, y_pred=Yguess,
                                       average='weighted'))
    print("Recall", recall_score(y_true=Ytest, y_pred=Yguess,
                                 average='weighted'))
    print("F1-score", f1_score(y_true=Ytest, y_pred=Yguess,
                                 average='weighted'))
    print()

    scores = precision_recall_fscore_support(Ytest, Yguess,
                                             labels=labels)
    print(pd.DataFrame(scores, columns=labels,
                       index=["Precision", "Recall", "F-score",
                              "Support"]).drop(["Support"]), '\n')

    # Print confusion matrix.
    matrix = confusion_matrix(Ytest, Yguess, labels=labels)
    print(pd.DataFrame(matrix, index=labels, columns=labels), '\n')
    
    # Calculate prior probabilities.
    print("Prior probability per class")
    counter = Counter([word for word in Ytest])

    for label, count in counter.items():
        prior_proba = count / len(Ytest)
        print(label, prior_proba)
    print()
        
    print("Posterior probability per class")
    for label, count in counter.items():
        # calculate prior probability
        prior_proba = count / len(Ytest)

        # determine values from confusion matrix
        false_pos = matrix.sum(axis=0) - np.diag(matrix)
        false_neg = matrix.sum(axis=1) - np.diag(matrix)
        true_pos = np.diag(matrix)
        true_neg = matrix.sum() - (false_pos + false_neg + true_pos)

        # calculate posterior probability
        posterior_proba = (true_pos * prior_proba) /\
                          ((true_pos * prior_proba) +
                           ((1 - prior_proba) * true_neg))
        print(label, max(posterior_proba))


if __name__ == '__main__':
    main()
