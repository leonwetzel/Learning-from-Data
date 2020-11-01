#!/usr/bin/env python3
import os
from tqdm import tqdm
import requests
import gzip


def download_embeddings(url="https://s3.amazonaws.com/dl4j-distribution/" \
                        "GoogleNews-vectors-negative300.bin.gz",
                        file_name="GoogleNews-vectors-negative300.bin.gz"):
    """
    Downloads Google News vectors and extracts the
    information.
    :return:
    """
    print("Downloading GoogleNewsVectors...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("Download failed, something went wrong!\n")
    else:
        print("Download completed!\n")


def decompress(archive_name, destination_name):
    """
    Decompresses a given .gz file.
    :param destination_name:
    :param archive_name:
    :return:
    """
    print(f"Decompressing {archive_name}...")
    fp = open(f"{destination_name}", "wb")
    with gzip.open(archive_name, 'rb') as f:
        file_content = f.read()
    fp.write(file_content)
    fp.close()
    os.remove(archive_name)
    print(f"Decompression of {archive_name} is finished!\n")
