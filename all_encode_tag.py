from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer, util
import time
import torch
import os

def get_training_job_rank_world():
    hosts = os.environ.get("TOTAL_WORKERS")
    current_host = os.environ.get("JOB_COMPLETION_INDEX")
    if hosts is None or current_host is None:
        return 0, 1
    hosts = int(hosts)
    current_host = int(current_host)
    return current_host, hosts

# Get rank and world size
rank, world_size = get_training_job_rank_world()

# Prepare the data
dataset = "MSMARCO"
data_path = '/data/richard/taggerv2/test/test6/beir/outputs/datasets/msmarco'
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
one_section_size = len(corpus) // world_size
start = one_section_size * rank

input_docs = list(corpus.values())[start:start + one_section_size]
# input_docs = list(corpus.values())[:5000]

print(f'Current_id: {start} to {start + one_section_size}')

#Load the model
# model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')    # TAS-B
# model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v2')     # SBERT
model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device='cuda:0')      # gtr-t5-xl
# model = SentenceTransformer("./local-gtr-t5-xl-onnx", backend="onnx")
# model = SentenceTransformer('BAAI/bge-large-en-v1.5')       # BGE

model = model.half()

print("Start embedding docs:")
start = time.perf_counter()       # high-resolution timer
doc_emb = model.encode(input_docs)
end = time.perf_counter()
print(doc_emb.shape)
print(f"Took {end - start:.6f} seconds")

save_dir = '/data/richard/taggerv2/test/test6/beir/outputs/inference/gtr-t5-xl_orig/'
os.makedirs(save_dir, exist_ok=True)
torch.save(doc_emb, f'{save_dir}/corpus_{rank}.pth')
