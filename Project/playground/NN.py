#!/usr/bin/env python3
import os

import pandas as pd
import numpy as np

from gensim.models import KeyedVectors

from assets import download_embeddings, decompress

EMBEDDINGS_FILE_NAME = "GoogleNews-vectors-negative300.bin"


def main():
    """
    Another demo script for performing
    COP Political Orientation Classification
    (COPPOC).
    :return:
    """
    if os.path.isfile(f"{EMBEDDINGS_FILE_NAME}.gz"):
        # file is present, but compressed
        decompress(archive_name=f"{EMBEDDINGS_FILE_NAME}.gz",
              destination_name=EMBEDDINGS_FILE_NAME)
    elif not os.path.isfile(f"{EMBEDDINGS_FILE_NAME}"):
        # file is not present
        download_embeddings()
        decompress(archive_name=f"{EMBEDDINGS_FILE_NAME}.gz",
              destination_name=EMBEDDINGS_FILE_NAME)

    print(f"Loading embeddings from {EMBEDDINGS_FILE_NAME}...")
    embeddings = KeyedVectors.load_word2vec_format(
        fname=f"{EMBEDDINGS_FILE_NAME}", binary=True
    )

    with open("../COP_filt3_sub/filtered_data.json", "r") as F:
        data = pd.read_json(F)

    print(data[["headline"]].head())


if __name__ == '__main__':
    main()
