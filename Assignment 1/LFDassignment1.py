import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

def read_corpus(corpus_file, use_sentiment):
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

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])

    return documents, labels


def identity(x):
    """
    A dummy function that just returns its input.
    :param x:
    :return:
    """
    return x


if __name__ == '__main__':
    """
    This script reads text from a given
    file and predicts either the sentiment type
    or the genre.
    """
    if sys.argv[1] == "-s":
        use_sentiment = True
    else:
        use_sentiment = False

    X, Y = read_corpus('trainset.txt', use_sentiment)
    split_point = int(0.75 * len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]
    labels = np.unique(Ytest)
    # let's use the TF-IDF vectorizer
    tfidf = True

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity,
                              tokenizer=identity)
    else:
        vec = CountVectorizer(preprocessor=identity,
                              tokenizer=identity)

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),
                           ('cls', MultinomialNB())])

    # Trains the classifier, by feeding documents (X)
    # and labels (y).
    classifier.fit(Xtrain, Ytrain)

    # Classifier makes a prediction, based on
    # a test set of documents.
    Yguess = classifier.predict(Xtest)

    # Prints the scores.
    print(accuracy_score(y_true=Ytest, y_pred=Yguess))
    scores = precision_recall_fscore_support(Ytest, Yguess, labels=labels)
    print(pd.DataFrame(scores, columns=labels, index=["Precision","Recall","F-score","Support"]))
    # Print confusion matrix.
    confusion_matrix = confusion_matrix(Ytest, Yguess, labels=labels)
    print(pd.DataFrame(confusion_matrix, index=labels, columns=labels))
