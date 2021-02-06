import csv
import json
from typing import Dict

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