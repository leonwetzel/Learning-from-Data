import sys
from collections import Counter

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


def main():
    """
    This script reads text from a given
    file and predicts either the sentiment type
    or the genre, based on a provided flag setting '-s'.
    """
    try:
        if sys.argv[1] == "-s":
            use_sentiment = True
        else:
            print("Invalid flag! Sentiment will not be used.")
            use_sentiment = False
    except IndexError:
        # catch cases where no flag is given
        use_sentiment = False

    X, Y = read_corpus('trainset.txt', use_sentiment)
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

    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),
                           ('cls', MultinomialNB())])

    # Trains the classifier, by feeding documents (X)
    # and labels (y).
    classifier.fit(Xtrain, Ytrain)

    # Classifier makes a prediction, based on
    # a test sample of documents.
    Yguess = classifier.predict(Xtest)

    # Prints the scores.
    print("Overall accuracy", accuracy_score(y_true=Ytest, y_pred=Yguess), '\n')
    scores = precision_recall_fscore_support(Ytest, Yguess, labels=labels)
    print(pd.DataFrame(scores, columns=labels, index=["Precision", "Recall", "F-score", "Support"]).drop(["Support"]), '\n')

    # Print confusion matrix.
    matrix = confusion_matrix(Ytest, Yguess, labels=labels)
    print(pd.DataFrame(matrix, index=labels, columns=labels), '\n')
    
    # Calculate prior probabilities.
    counter = Counter()
    for word in Ytest:
        counter[word] += 1

    print("Prior probability per class")
    for label, count in counter.items():
        prior_proba = count / len(Ytest)
        print(label, prior_proba)
    print()

    # Calculate posterior probabilities
    print("Posterior probability per class")
    for label, log_prior in zip(classifier[1].classes_, classifier[1].feature_log_prob_):
        print(label, log_prior)


if __name__ == '__main__':
    main()
