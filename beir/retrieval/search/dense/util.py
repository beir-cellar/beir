import torch
import numpy as np
import csv

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
    return a/np.linalg.norm(a, ord=2, axis=1, keepdims=True)

def save_dict_to_tsv(_dict, output_path, keys=[]):
    
    with open(output_path, 'w') as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if keys: writer.writerow(keys)
        for key, value in _dict.items():
            writer.writerow([key, value])

def load_tsv_to_dict(input_path, header=True):
    
    mappings = {}
    reader = csv.reader(open(input_path, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    if header: next(reader)
    for row in reader: 
        mappings[row[0]] = int(row[1])
    
    return mappings