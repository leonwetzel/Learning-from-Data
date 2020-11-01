# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Evaluating COPPOC performances
#
# In this Jupyter Notebook, we examine the performances of the COP
#  Political Orientation Classifier (COPPOC). We see the results for the
#   classifier that uses body texts and the classifiers that use the
#    article headlines.

# + pycharm={"name": "#%%\n"}
import os
import json
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tabulate

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    f1_score, recall_score, precision_score, accuracy_score

# + pycharm={"name": "#%%\n"}
from data_management import flatten, get_political_orientation

DATA_DIR = "COP_filt3_sub"
TEST_DATA = "test.json"
SCORE_TYPE = "macro"
MODEL_DIR = r"demo/models"

try:
    with open(f"{DATA_DIR}/{TEST_DATA}", "r") as F:
        test_data = json.load(F)
        test_data = flatten(test_data)
except FileNotFoundError:
    response = requests.get(
        "https://teaching.stijneikelboom.nl/lfd2021/test/COP25.filt3.sub.json",
        stream=True)

    with open(f"{DATA_DIR}/{TEST_DATA}", "w") as F:
        F.write(response.content)

test_data = pd.DataFrame(test_data)

plt.style.use('default')
# -

# ## Quick observations
#
# Let's take a look at the provided test data.
#

# + pycharm={"name": "#%%\n"}
counts = test_data['newspaper'].value_counts()

ax = counts.plot.bar("newspaper", rot=90,
                      title="Distribution of newspapers",
                      xlabel="Newspaper", ylabel="Amount of articles",
                      color="blue")
# -

# The distribution of articles per newspaper does not seen to be equal among all the newspapers.
#

# + pycharm={"name": "#%%\n"}
counts = test_data['political_orientation'].value_counts()

ax = counts.plot.bar(x='political_orientation', y="newspaper", rot=0,
                      title="Distribution of newspapers",
                      xlabel="Political orientation", ylabel="Amount of articles",
                      color="blue")

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
# -

# ## Using body texts
#
# During development of COPPOC, we trained a linear Support Vector Machine (SVM) model that classifies
# news articles using their body text. This model uses an L1 penalty,
#  the `hinge` loss function and the `crammer singer` multi-class strategy.
#   The model is configured to solve a dual optimization problem and to
#    calculate the model intercept, using a
#    regularization parameter $C$ with a value of 0.7, a stopping criteria
#     tolerance of 0.0001, an intersept scale of 1 and a maximum number of
#      iterations of 1000. Custom class weights were not used.

# + pycharm={"name": "#%%\n"}
pipeline = joblib.load("classification_pipeline.joblib")

y_test = test_data['political_orientation']
labels = np.unique(y_test)
y_guess = pipeline.predict(test_data['body'])

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
print(pd.DataFrame(matrix, index=labels, columns=labels))
# -

# ## Using headlines
#
# It turned out that using headlines from news articles is faster and
# more computer-friendly than the body texts. We experimented with two
# variants of Naive Bayes (Multinomial and Bernoulli), a Stochastic
# Gradient Descent model (SGD) and a Support Vector Machine (SVM) model.

# ### Naive Bayes
#
# The Multinomial Naive Bayes and Bernoulli Naive Bayes are fitting for
# solving the classification problem, as they are designed to cope
# with our data formats.

# + pycharm={"name": "#%%\n"}
pipeline = joblib.load(f"{model_directory}/nb_multinomial.joblib")

y_test = test_data['political_orientation']
labels = np.unique(y_test)
y_guess = pipeline.predict(test_data['headline'])

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
print(pd.DataFrame(matrix, index=labels, columns=labels))

# + [markdown] pycharm={"name": "#%% md\n"}
# We can observe that precision and recall scores are quite similar to
# each other, for both the left-sided and the right-sided orientations.
#

# + pycharm={"name": "#%%\n"}
pipeline = joblib.load(f"{model_directory}/nb_bernoulli.joblib")

y_test = test_data['political_orientation']
labels = np.unique(y_test)
y_guess = pipeline.predict(test_data['headline'])

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
print(pd.DataFrame(matrix, index=labels, columns=labels))
# -

# ### Stochastic Gradient Descent
#
# We can observe that

# + pycharm={"name": "#%%\n"}
pipeline = joblib.load(f"{model_directory}/sgd.joblib")

y_test = test_data['political_orientation']
labels = np.unique(y_test)
y_guess = pipeline.predict(test_data['headline'])

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
print(pd.DataFrame(matrix, index=labels, columns=labels))
# -

# ### Support Vector Machine
#
# We can observe that

# + pycharm={"name": "#%%\n"}
pipeline = joblib.load(f"{model_directory}/svm.joblib")

y_test = test_data['political_orientation']
labels = np.unique(y_test)
y_guess = pipeline.predict(test_data['headline'])

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
print(pd.DataFrame(matrix, index=labels, columns=labels))
