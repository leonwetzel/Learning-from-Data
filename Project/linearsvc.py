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

# # Exploring the Lexis Nexis data
#
# In this notebook, we take a look at the data for the Learning from Data project. This data is exported from Lexis Nexis by Stijn Eikelboom. We use his exported data for analysis.
#
# ## Loading the data
#

# +
# general dependencies
import sys
import os

from collections import defaultdict
from collections import Counter
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# custom scripts
import data_management as dm


pd.set_option('display.max_rows', None)

print(f"Python version: {sys.version}")
print(f"Current working dir: {os.getcwd()}")

DATA_DIRECTORY = f"{os.getcwd()}/COP_filt3_sub"
# -

# Next to having downloaded the files, we have also merged them into a
# single file. That will save us time and effort in the future...

# Let's look into the data!

# + pycharm={"name": "#%%\n"}
data = pd.read_json(f"{DATA_DIRECTORY}/filtered_data.json")
# -

data.info()

# Types aren't correct, cop_edition has to be integer, collection_start and collection_end datetime.

data["collection_start"] = pd.to_datetime(data["collection_start"])

data.info()

data["collection_end"] = pd.to_datetime(data["collection_end"])

data["cop_edition"] = data["cop_edition"].replace("6a","6")

data["cop_edition"] = data["cop_edition"].astype(int)

data.info()

data["newspaper"].value_counts()

# +
australia = ["The Australian", "Sydney Morning Herald (Australia)", "The Age (Melbourne, Australia)"]
india = ["The Times of India (TOI)", "The Hindu"]
south_africa = ["The Times (South Africa)","Mail & Guardian"]
united_states = ["The New York Times","The Washington Post"]

def assign_country(newspaper):
    if newspaper in australia:
        return "Australia"
    elif newspaper in india:
        return "India"
    elif newspaper in south_africa:
        return "South Africa"
    else:
        return "United States"
    
data["country"] = data["newspaper"].apply(assign_country)
data["country"].value_counts()

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(data['body'])
X = tfidf_vect.transform(data['body'])
y = data['political_orientation']
scores = []
classifier = svm.LinearSVC()
cv = KFold(n_splits=10, shuffle=False)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    classifier.fit(X_train, y_train)
    scores.append(classifier.score(X_test, y_test))
print(sum(scores) / len(scores))
# -


