"""
This code shows how to generate using parallel GPU's for very long corpus.
Multiple GPU's can be used to generate faster!

We use torch.multiprocessing module and define multiple pools for each GPU.
Then we chunk our big corpus into multiple smaller corpus and generate simultaneously.

Important to use the code within the __main__ module!

Usage: CUDA_VISIBLE_DEVICES=0,1 python query_gen_multi_gpu.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel

import pathlib, os
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    #### Download scifact.zip dataset and unzip the dataset
    dataset = "trec-covid"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus = GenericDataLoader(data_path).load_corpus()

    ###########################
    #### Query-Generation  ####
    ###########################

    #Define the model
    model = QGenModel("BeIR/query-gen-msmarco-t5-base-v1")

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    generator = QGen(model=model)

    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    prefix = "gen-3"

    #### Generating 3 questions per document for all documents in the corpus 
    #### Reminder the higher value might produce diverse questions but also duplicates
    ques_per_passage = 3

    #### Generate queries per passage from docs in corpus and save them in original corpus
    #### check your datasets folder to find the generated questions, you will find below:
    #### 1. datasets/scifact/gen-3-queries.jsonl
    #### 2. datasets/scifact/gen-3-qrels/train.tsv

    chunk_size = 5000    # chunks to split within each GPU
    batch_size = 64       # batch size within a single GPU 

    generator.generate_multi_process(
        corpus=corpus, 
        pool=pool, 
        output_dir=data_path, 
        ques_per_passage=ques_per_passage, 
        prefix=prefix, 
        batch_size=batch_size)
    
    # #Optional: Stop the proccesses in the pool
    # model.stop_multi_process_pool(pool)