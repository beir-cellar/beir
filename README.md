
<!-- <h1>
<img style="vertical-align:middle" width="120" height="120" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo.png" />
BeIR: A Heterogeneous Benchmark for IR
</h1> -->

<!-- <h1 text-align= "center">
    <img width="300" height="120" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" style="vertical-align: middle;"/>
</h1> -->

<h1 style="text-align:center">
<img style="vertical-align:middle" width="450" height="180" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" />
</h1>

<!-- <h3 align="center">
BEIR: A heterogeneous benchmark for Information Retrieval
</h3> -->


![PyPI](https://img.shields.io/pypi/v/beir)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Nthakur20/StrapDown.js/graphs/commit-activity)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchmarkir/beir/blob/main/examples/retrieval/Retrieval_Example.ipynb)
[![Downloads](https://pepy.tech/badge/beir)](https://pepy.tech/project/beir)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/benchmarkir/beir/)

## What is it?

**BEIR** consists a **heterogeneous benchmark** for diverse sentence or passage IR level tasks. It also provides a **common and easy framework** for evaluation of your NLP models on them.

The package takes care of the downloading, hosting, preprocessing datasets and providing you in a single easy to understand dataset zip folders. We take care of transforming the dataset and provide 15 diverse datasets used for IR in the both academia and industry, with more to add. Further the package provides an easy framework to evalaute your models against some competitive benchmarks including Sentence-Transformers (SBERT), Dense Passage Retrieval (DPR), Universal Sentence Encoder (USE-QA) and Elastic Search.

### Worried about your dataset or model not present in the benchmark?

Worry not! You can easily add your dataset into the benchmark by following this data format (here) and also you are free to evaluate your own model and required to return a dictionary with mappings (here) and you can evaluate your IR model using our easy plugin code.

Want us to add a new dataset or a new model? feel free to post an issue here or make a pull request!

## Table Of Contents

- [Installation](https://github.com/UKPLab/beir#installation)
- [Getting Started](https://github.com/UKPLab/beir#getting-started)
    - [Quick Example](https://github.com/UKPLab/beir#quick-example)
    - [Google Colab](https://colab.research.google.com/github/benchmarkir/beir/blob/main/examples/retrieval/Retrieval_Example.ipynb)
    - [Evaluate on a Custom Dataset?](https://github.com/UKPLab/beir#evaluate-on-a-custom-dataset)
    - [Evaluate your own Custom Model?](https://github.com/UKPLab/beir#evaluate-your-own-custom-model)
- [Examples](https://github.com/UKPLab/beir#examples)
    - [Retrieval](https://github.com/UKPLab/beir#retrieval)
    - [Generation](https://github.com/UKPLab/beir#generation)
    - [Filtration](https://github.com/UKPLab/beir#)
- [Datasets](https://github.com/UKPLab/beir#datasets)
- [Benchmarking](https://github.com/UKPLab/beir#benchmarking)
- [Citing & Authors](https://github.com/UKPLab/beir#citing--authors)
## Installation

Install via pip:

```python
pip install beir
```

If you want to build from source, use:

```python
$ git clone https://github.com/benchmarkir/beir.git
$ pip install -e .
```

Tested with python versions 3.6 and 3.7

## Getting Started

### Google Colab Example

Try it out live with our [Google Colab Example](https://colab.research.google.com/github/benchmarkir/beir/blob/main/examples/retrieval/Retrieval_Example.ipynb).

### Quick Example

```python
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "nq.zip"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

model = DRES(models.SentenceBERT("distilroberta-base-msmarco-v2"))
retriever = EvaluateRetrieval(model)

results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```

### Evaluate on your Custom Dataset?

Load your custom corpus, query, and qrels as python ``dict`` in the format shown below:

```python
#### Corpus ####
corpus = {
    "doc1" : {
        "title": "Albert Einstein", 
        "text": "Albert Einstein was a German-born theoretical physicist. who developed the theory of relativity, \
                 one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for \
                 its influence on the philosophy of science. He is best known to the general public for his mass–energy \
                 equivalence formula E = mc2, which has been dubbed 'the world's most famous equation'. He received the 1921 \
                 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law \
                 of the photoelectric effect', a pivotal step in the development of quantum theory."
        },
    "doc2" : {
        "title": "", # Keep title an empty string if not present
        "text": "Wheat beer is a top-fermented beer which is brewed with a large proportion of wheat relative to the amount of \
                 malted barley. The two main varieties are German Weißbier and Belgian witbier; other types include Lambic (made\
                 with wild yeast), Berliner Weisse (a cloudy, sour beer), and Gose (a sour, salty beer)."
    },
}

#### Queries #### 
queries = {
    "q1" : "Who developed the mass-energy equivalence formula?",
    "q2" : "Which beer is brewed with a large proportion of wheat?"
}

#### Qrels #### 
qrels = {
    "q1" : {"doc1": 1},
    "q2" : {"doc2": 1},
}
```

### Evaluate your own Custom Model?

Mention your custom model in a class and have two functions: 1. ``encode_queries`` and 2. ``encode_corpus``. 

```python
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

class YourCustomModel:
    def __init__(self, model_path=None, **kwargs)
        self.model = None # ---> HERE Load your custom model
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        pass
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        pass

custom_model = DRES(YourCustomModel(model_path="your-custom-model-path"))
```

## Examples

For all examples, see below:

### All in One
- [Google Colab Example](https://colab.research.google.com/github/UKPLab/beir/blob/main/examples/retrieval/Retrieval_Example.ipynb)

### Retrieval
- [BM25 Retrieval using Elasticsearch](https://github.com/UKPLab/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_bm25.py)
- [Exact Search Retrieval using Dense Model](https://github.com/UKPLab/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_dense.py)
- [Faiss Search Retrieval using Dense Model](https://github.com/UKPLab/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_faiss_dense.py)
- [Training Dense Retrieval Model](https://github.com/UKPLab/beir/blob/main/examples/retrieval/training/train_dense.py)
- [Custom Dataset Retrieval Evaluation](https://github.com/UKPLab/beir/blob/main/examples/retrieval/evaluation/custom/evaluate_custom_dataset.py)
- [Custom Model Retrieval Evaluation](https://github.com/UKPLab/beir/blob/main/examples/retrieval/evaluation/custom/evaluate_custom_model.py)

### Generation
- [Question Generation using T5 Seq2Seq model](https://github.com/UKPLab/beir/blob/main/examples/generation/query_gen.py)
- [Question Generation and Zero-Shot Dense Encoder Training](https://github.com/UKPLab/beir/blob/main/examples/generation/query_gen_and_train.py)

### Filtration
- [Question Generation and Filtration using Tiny-BERT Cross-Encoder](https://github.com/UKPLab/beir/blob/main/examples/filtration/query_gen_and_filter.py)
- [Question Generation and Filtration and Zero-shot Dense Encoder Training](https://github.com/UKPLab/beir/blob/main/examples/filtration/query_gen_filter_and_train.py)

## Datasets

Available datasets include:

- [TREC-COVID](https://ir.nist.gov/covidSubmit/index.html)
- [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
- [NQ](https://ai.google.com/research/NaturalQuestions)
- [HotpotQA](https://hotpotqa.github.io/)
- [NewsQA](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)
- [FiQA](https://sites.google.com/view/fiqa/home)
- [ArguAna](http://argumentation.bplaced.net/arguana/data)
- [Touche-2020](https://webis.de/events/touche-20/)
- [CQaDupstack](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)
- [Quora](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
- [DBPedia-v2](https://iai-group.github.io/DBpedia-Entity/)
- [SCIDOCS](https://allenai.org/data/scidocs)
- [FEVER](https://fever.ai/)
- [Climate-FEVER](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)
- [Signal-1M](https://research.signal-ai.com/datasets/signal1m-tweetir.html) (Optional)
- [BioASQ](http://bioasq.org/) (Optional)

## Data Formats

```python
from beir.datasets.data_loader import GenericDataLoader

data_path = "datasets/trec-covid/"
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Corpus
for doc_id, doc_metadata in corpus.items():
    print(doc_id, doc_metadata)
# ug7v899j  {"title": "Clinical features of culture-proven Mycoplasma...", "text": "This retrospective chart review describes the epidemiology..."}
# 02tnwd4m  {"title": "Nitric oxide: a pro-inflammatory mediator in lung disease?, "text": "Inflammatory diseases of the respiratory tract are commonly associated..."}
# ...

# Queries
for query_id, query_text in query.items():
    print(query_id, query_text)
# 1     what is the origin of COVID-19?
# 2     how does the coronavirus respond to changes in the weather?
# ...

# Query Relevance Judgements (Qrels)
for query_id, metadata in qrels.items():
    for doc_id, gold_score in metadata.items():
        print(query_id, doc_id, gold_score)
# 1     005b2j4b    2
# 1     00fmeepz    1
# ...
```

## Benchmarking

The benchmarking results will be included soon.

<!-- The Table shows the NDCG@10 scores.

| Domain     |Dataset       | BM25    | SBERT   | USE-QA  | DPR     |
| :---------:|------------: |:------: |:------: |:------: |:------: |
|            | TREC-COVID   | 0.616   | 0.461   |         |         |
| Bio-Medical| BioASQ       |         |         |         |         |
|            | NFCorpus     | 0.294   | 0.233   |         |         |
|            |              |         |         |         |         |
| Question   | NQ           | 0.481   | 0.530   |         |         |
| Answering  | HotpotQA     | 0.601   | 0.419   |         |         |
|            |              |         |         |         |         |
| News       | NewsQA       | 0.457   | 0.263   |         |         |
|            |              |         |         |         |         |
| Twitter    | Signal-1M    |  0.477  | 0.272   |         |         |
|            |              |         |         |         |         |
| Finance    | FiQA-2018    |         |  0.223  |         |         |
| Argument   | ArguAna      |  0.441  |  0.415  |         |         |
|            | Touche-2020  |  0.605  |         |         |         |
|            |              |         |         |         |         |
| Duplicate  | CQaDupstack  |  0.069  |  0.061  |         |         |
| Question   | Quora        |         |         |         |         |
|            |              |         |         |         |         |
|  Entity    | DBPedia-v2   |  0.285  |  0.261  |         |         |
|            |              |         |         |         |         |
| Scientific | SCIDOCS      |         |         |         |         |
|            |              |         |         |         |         |
| Claim      | FEVER        |  0.649  |  0.601  |         |         |
|Verification|Climate-FEVER |  0.179  |  0.192  |         |         | -->


## Citing & Authors

The main contributors of this repository are:
- [Nandan Thakur](https://github.com/Nthakur20), Personal Website: [https://nthakur.xyz](https://nthakur.xyz)

Contact person: Nandan Thakur, [nandant@gmail.com](mailto:nandant@gmail.com)

[https://www.ukp.tu-darmstadt.de/](https://www.ukp.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

