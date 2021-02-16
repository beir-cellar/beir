from typing import Dict
import csv
import json
import logging
import os
import requests
import zipfile

logger = logging.getLogger(__name__)

def download_url(url: str, save_path: str, chunk_size: int = 128):
    if not os.path.isfile(save_path):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

def unzip(zip_file: str, out_dir: str):
    if not os.path.isdir(zip_file.replace(".zip", "")):
        zip_ = zipfile.ZipFile(zip_file, "r")
        zip_.extractall(path=out_dir)
        zip_.close()

def download_and_unzip(url: str, out_dir: str) -> str:
    
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)
    
    logger.info("Downloading {} ...".format(dataset))
    download_url(url, zip_file)
    
    logger.info("Unzipping {} ...".format(dataset))
    unzip(zip_file, out_dir)
    
    return os.path.join(out_dir, dataset.replace(".zip", ""))

def write_to_json(output_file: str, data: Dict[str, str]):
    with open(output_file, 'w') as fOut:
        for idx, text in data.items():
            json.dump({
                "_id": idx, 
                "text": text,
                "metadata": {}
            }, fOut)
            fOut.write('\n')

def write_to_tsv(output_file: str, data: Dict[str, str]):
    with open(output_file, 'w') as fOut:
        writer = csv.writer(fOut, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["query-id", "corpus-id", "score"])
        for query_id, corpus_dict in data.items():
            for corpus_id, score in corpus_dict.items():
                writer.writerow([query_id, corpus_id, score])
