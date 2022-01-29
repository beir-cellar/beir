from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import PassageExpansion as PassageExp
from beir.generation.models import TILDE

import pathlib, os
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus = GenericDataLoader(data_path).load_corpus()

#############################
#### TILDE Model Loading ####
#############################

#### Model Loading 
model_path = "ielab/TILDE"
generator = PassageExp(model=TILDE(model_path))

#### TILDE passage expansion using top-k most likely expansion tokens from BERT Vocabulary ####
#### Only supports bert-base-uncased (TILDE) model for now 
#### Prefix is required to store the final expanded passages as a corpus.jsonl file 
prefix = "tilde-exp"

#### Expand useful tokens per passage from docs in corpus and save them in a new corpus
#### check your datasets folder to find the expanded passages appended with the original, you will find below:
#### 1. datasets/scifact/tilde-exp-corpus.jsonl

#### Batch size denotes the number of passages getting expanded at once
batch_size = 64

#### top-k value will retrieve the top-k expansion terms with highest softmax probability 
#### These tokens are individually appended once to the passage
#### We remove stopwords, bad-words (punctuation, etc.) and words in original passage.  
top_k = 200

generator.expand(corpus, output_dir=data_path, prefix=prefix, batch_size=batch_size, top_k=top_k)