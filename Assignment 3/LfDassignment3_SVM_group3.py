import sys
import os

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, \
    precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
stop_words = stopwords.words('english')


def read_corpus(corpus_file):
    """
    Reads information from a given file, using sentiment if indicated.
    We assume a provided file consists of 1) a genre, 2) a sentiment type
    and 3) a review.
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

            labels.append(tokens[1])

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

    labels = np.unique(Ytest)

    vec = TfidfVectorizer(preprocessor=identity,
                            tokenizer=identity, lowercase=True,
                            analyzer='word', ngram_range=(1,3),
                            stop_words=None)

    svc = SVC(kernel='linear', C=10, degree=1, gamma=1,
              shrinking=True, probability=True,
              decision_function_shape='ovr')

    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),
                         ('clf', svc)])

    # with thanks to
    # - https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
    # parameters = {
    #     'vec__lowercase': [True],
    #     'vec__analyzer': ['word'],
    #     'vec__ngram_range': [(1,1), (1,2), (1,3)],
    #     'vec__stop_words': [None],
    #     'clf__kernel': ['rbf', 'linear'],
    #     'clf__C': [0.7, 10],
    #     'clf__degree': [1],
    #     'clf__gamma': [1.0],
    #     'clf__shrinking': [True],
    #     'clf__probability': [True],
    #     'clf__decision_function_shape': ['ovr'],
    # }

    # set up scorer, so we can compare both F1 and accuracy scores
    # scoring = {'F1': make_scorer(f1_score, average='weighted'),
    #            'Accuracy': make_scorer(accuracy_score)}
    #
    # classifier = GridSearchCV(pipeline, parameters,
    #                           scoring=scoring,
    #                           n_jobs=-1, cv=2,
    #                           return_train_score=False,
    #                           refit='F1',
    #                           verbose=3, error_score=0.0,
    #                           pre_dispatch='2*n_jobs')

    # Trains the classifier, by feeding documents (X)
    # and labels (y).
    classifier.fit(Xtrain, Ytrain)

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

    # print(f"Best parameter combination: {classifier.best_params_}")


if __name__ == '__main__':
    main()
