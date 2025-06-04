from __future__ import annotations

import csv
import os
import pickle

import numpy as np
import torch


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
    return torch.mm(a_norm, b_norm.transpose(0, 1))  # TODO: this keeps allocating GPU memory


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


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, ord=2, axis=1, keepdims=True)


def save_dict_to_tsv(_dict, output_path, keys=[]):
    with open(output_path, "w") as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if keys:
            writer.writerow(keys)
        for key, value in _dict.items():
            writer.writerow([key, value])


def load_tsv_to_dict(input_path, header=True):
    mappings = {}
    reader = csv.reader(open(input_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    if header:
        next(reader)
    for row in reader:
        mappings[row[0]] = int(row[1])

    return mappings


def save_embeddings(
    embeddings: np.ndarray | list[torch.Tensor], text_ids: list[str], output_filename: str = "./embeddings/"
):
    """
    Saves the embeddings to a pickle file.
    :param embeddings: The embeddings to save.
    :param output_path: The path where the embeddings will be saved.
    """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if isinstance(embeddings[0], torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()  # Convert to numpy array if it's a tensor

    with open(output_filename, "wb") as f:
        pickle.dump((embeddings, text_ids), f)


def pickle_load(path: str) -> tuple[np.ndarray, list[str]]:
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup
