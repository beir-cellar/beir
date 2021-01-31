from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from QueGenerator import QueGenerator

import torch
import tqdm
import random
import csv
import json
import argparse
import gzip
import io
import logging
import itertools
import pathlib, os
import pdb

os.environ["CUDA_VISIBLE_DEVICES"]="5"

# Set Seed
def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Load data
data_path = "/home/ukp/srivastava/projects/beir/beir/datasets/trec-covid"
corpus, _, _ = GenericDataLoader(data_path).load(split="test")

# def getAllText(my_dict):
#     return my_dict["title"]+" "+my_dict["text"]

# corpus_ids = list(corpus.keys())[:100]
# corpus = [getAllText(corpus[cid]) for cid in corpus_ids]

output_dir = "/home/ukp/srivastava/projects/sbert_retriever/datasets/trec-covid/new-models-old-format/msmarco/"
ques_per_passage = 5
batch_size = 2

model_source_path = "/home/ukp/srivastava/projects/generation-train/output/msmarco/"
models = ["bart-base-1-epoch-default-lr/checkpoint-66500", "t5-base-1-epoch-default-lr/checkpoint-66500", "t5-base-2-epoch-default-lr/checkpoint-133000"]

for model in models:
    model_path = model_source_path + model
    file_prefix = model.split("/")[0]+"_"
    qgen = QueGenerator("general",model_path=model_path)
    qgen.generate(corpus, output_dir, ques_per_passage, batch_size, file_prefix)

# model_path = "/home/ukp/srivastava/projects/generation-train/output/msmarco/"
# file_prefix = model.split("/")[0]+"_"
# qgen = QueGenerator("general",model_path=model_path)
# qgen.generate(corpus, output_dir, ques_per_passage, batch_size, file_prefix)
