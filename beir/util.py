from __future__ import annotations

import csv
import json
import logging
import os
import zipfile

import requests
import torch
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with (
        open(save_path, "wb") as fd,
        tqdm(
            desc=save_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar,
    ):
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()


def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info(f"Downloading {dataset} ...")
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info(f"Unzipping {dataset} ...")
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))


def write_to_json(output_file: str, data: dict[str, str]):
    with open(output_file, "w") as fOut:
        for idx, meta in data.items():
            if isinstance(meta, str):
                json.dump({"_id": idx, "text": meta, "metadata": {}}, fOut)

            elif isinstance(meta, dict):
                json.dump(
                    {
                        "_id": idx,
                        "title": meta.get("title", ""),
                        "text": meta.get("text", ""),
                        "metadata": {},
                    },
                    fOut,
                )
            fOut.write("\n")


def write_to_tsv(output_file: str, data: dict[str, str]):
    with open(output_file, "w") as fOut:
        writer = csv.writer(fOut, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["query-id", "corpus-id", "score"])
        for query_id, corpus_dict in data.items():
            for corpus_id, score in corpus_dict.items():
                writer.writerow([query_id, corpus_id, score])


def save_runfile(
    output_file: str,
    results: dict[str, dict[str, float]],
    run_name: str = "beir",
    top_k: int = 1000,
):
    with open(output_file, "w") as fOut:
        for qid, doc_dict in results.items():
            sorted_docs = sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)[:top_k]
            for doc_id, score in sorted_docs:
                fOut.write(f"{qid} Q0 {doc_id} 0 {score} {run_name}\n")


def load_runfile(input_file: str) -> dict[str, dict[str, float]]:
    results = {}
    with open(input_file, encoding="utf-8") as fIn:
        for line in fIn:
            qid, _, doc_id, _, score, _ = line.strip().split(" ")
            if qid not in results:
                results[qid] = {}
            results[qid][doc_id] = float(score)
    return results


def save_results(
    output_file: str,
    ndcg: dict[int, float],
    _map: dict[int, float],
    recall: dict[int, float],
    precision: dict[int, float],
    mrr: dict[int, float] | None = None,
    recall_cap: dict[int, float] | None = None,
    hole: dict[int, float] | None = None,
):
    optional_names = ["mrr", "recall_cap", "hole"]

    with open(output_file, "w") as f:
        results = {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
        }

        # Add optional metrics
        for idx, metric in enumerate([mrr, recall_cap, hole]):
            if metric:
                results.update({optional_names[idx]: metric})

        json.dump(results, f, indent=4)

    logger.info(f"Saved evaluation results to {output_file}")
