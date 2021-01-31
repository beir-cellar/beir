import math
import torch

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunks_torch(dic, n):
    chunks = {key: list(torch.split(dic[key], n)) for key in dic.keys()}
    batches = math.ceil(len(list(dic.values())[0]) / n)
    for idx in range(batches):
        yield {key: value[idx] for key, value in chunks.items()}
    