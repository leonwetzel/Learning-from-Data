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

    # Load and split dataset
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
    print("Prior probability per class")
    counter = Counter([word for word in Ytest])

    for label, count in counter.items():
        prior_proba = count / len(Ytest)
        print(label, prior_proba)
    print()
        
    print("Posterior probability per class")
    """
    According to https://sebastianraschka.com/Articles/2014_naive_bayes_1.html:
    - posterior_proba = (conditional_proba * prior_proba) / evidence
    - For prior proba: see above!
    
    According to the lecture slides:
    - posterior_proba = p(c_j|i) =
     (p(i|c_j) * p(c_j)) /
     ([p(c_j) * p(i|c_j)] + [p(-c_j) * p(i|-c_j)])
    """
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
                          ((true_pos * prior_proba) + ((1 - prior_proba) * true_neg))
        print(label, max(posterior_proba))


if __name__ == '__main__':
    main()
