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

# + pycharm={"name": "#%%\n"}
dm.download()
dm.merge()
dm.filter_corpus()
# -

# Let's look into the data!

# + pycharm={"name": "#%%\n"}
data = pd.read_json(f"{DATA_DIRECTORY}/corpus.json")

#dat = pd.DataFrame([dm.flatten_json(x) for x in data["articles"]])

#print(dat.columns)
# -

# Look into newspaper distribution:

newspapers = []
for cop_year in data['articles']:
    for article in cop_year:
        newspapers.append(article['newspaper'])
newspaper_labels = np.unique(newspapers)
counter = Counter(newspapers)
for label, count in counter.items():
    relative_count = round(count/len(newspapers)*100, 2)
    print(label, 'absolute count:' ,count, 'relative count:', relative_count,'%')

# Look into political orientation distribution:

left_center = ['Sydney Morning Herald (Australia)','The Age (Melbourne, Australia)', 'The Hindu', 'Mail & Guardian', 'The New York Times', 'The Washington Post']
right_center = ['The Australian', 'The Times of India (TOI)', 'The Times (South Africa)']
left_count = 0
right_count = 0
for item in newspapers:
    if item in left_center:
        left_count += 1
    else:
        right_count += 1
print('number of articles with left-center orientation:', left_count, round(left_count/len(newspapers)*100,2), '%')
print('number of articles with right-center orientation:', right_count, round(right_count/len(newspapers)*100,2), '%')

# Look into popular topics for left and right:

# +
left_subjects = []
left_organizations = []
right_subjects = []
right_organizations = []
for cop_year in data['articles']:
    for article in cop_year:
        try:
            if article['newspaper'] in left_center:
                left_subjects.append(article['classification']['subject'][0]['name'])
                left_organizations.append(article['classification']['organization'][0]['name'])
            else:
                right_subjects.append(article['classification']['subject'][0]['name'])
                right_organizations.append(article['classification']['organization'][0]['name'])
        except TypeError:
            pass

print('Left subjects: ', Counter(left_subjects).most_common(15), '\n')
print('Left organizations: ', Counter(left_organizations).most_common(15), '\n')
print('Right subjects: ', Counter(right_subjects).most_common(15), '\n')
print('Right organizations: ', Counter(right_organizations).most_common(15), '\n')

# -


