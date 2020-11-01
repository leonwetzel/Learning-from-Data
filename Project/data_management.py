#!/usr/bin/env python3
import os
import json
import glob
import zipfile

import requests

URL = "https://teaching.stijneikelboom.nl/lfd2021/COP.filt3.sub.zip"
FILE_NAME = "COP.filt3.sub.zip"
DATA_DIR = "COP_filt3_sub"

political_orientations = {
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


def download():
    """
    Downloads the Lexis Nexis data from Stijn's website
    and puts in it the right directory
    """
    response = requests.get(URL, stream=True)

    file = open(FILE_NAME, 'wb')
    file.write(response.content)

    with zipfile.ZipFile(FILE_NAME, 'r') as zip_ref:
        zip_ref.extractall()

    file.close()
    os.remove(FILE_NAME)


def merge():
    """
    Merges data from the COP corpus into a single file.
    """
    result = []
    for f in glob.glob(f"{DATA_DIR}/COP*.json"):
        with open(f, "r") as infile:
            result.append(json.load(infile))

    with open(f"{DATA_DIR}/corpus.json", "w", encoding="utf-8") as outfile:
        json.dump(result, outfile)


def get_political_orientation(newspaper):
    """
    Wrapper function for retrieving the political
    orienation of a newspaper.
    :param newspaper:
    :return:
    """
    return political_orientations[newspaper]


def filter_corpus():
    """
    Gathers information from all the JSON files
    and only returns the relevant keys/fields for our
    project.
    :return:
    """
    result = []
    for f in glob.glob(f"{DATA_DIR}/COP*.json"):
        with open(f, "r") as infile:
            data = json.load(infile)
            articles = flatten(data)
            result.append(articles)

    with open(f"{DATA_DIR}/filtered_data.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2)


def flatten(data):
    """
    Flattens JSON from a given file.
    """
    articles = []
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
        articles.append(row)

    return articles
