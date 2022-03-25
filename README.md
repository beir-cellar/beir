<h1 style="text-align:center">
<img style="vertical-align:middle" width="450" height="180" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" />
</h1>

![PyPI](https://img.shields.io/pypi/v/beir)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Nthakur20/StrapDown.js/graphs/commit-activity)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing)
[![Downloads](https://pepy.tech/badge/beir)](https://pepy.tech/project/beir)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/benchmarkir/beir/)

> The development of BEIR benchmark is supported by:

<h3 style="text-align:center">
    <a href="http://www.ukp.tu-darmstadt.de"><img style="float: left; padding: 2px 7px 2px 7px;" width="220" height="100" src="./images/ukp.png" /></a>
    <a href="https://www.tu-darmstadt.de/"><img style="float: middle; padding: 2px 7px 2px 7px;" width="250" height="90" src="./images/tu-darmstadt.png" /></a>
    <a href="https://uwaterloo.ca"><img style="float: right; padding: 2px 7px 2px 7px;" width="320" height="100" src="./images/uwaterloo.png" /></a>
</h3>

<h3 style="text-align:center">
    <a href="https://huggingface.co/"><img style="float: middle; padding: 2px 7px 2px 7px;" width="400" height="80" src="./images/HF.png" /></a>
</h3>

## :beers: What is it?

**BEIR** is a **heterogeneous benchmark** containing diverse IR tasks. It also provides a **common and easy framework** for evaluation of your NLP-based retrieval models within the benchmark.

For more information, checkout our publications:

- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://openreview.net/forum?id=wCu6T5xFjeJ) (NeurIPS 2021, Datasets and Benchmarks Track)


## :beers: Table Of Contents

- [Installation](https://github.com/beir-cellar/beir#beers-installation)
- [Features](https://github.com/beir-cellar/beir#beers-features)
- [Leaderboard](https://github.com/beir-cellar/beir#beers-leaderboard)
- [Course Material on IR](https://github.com/beir-cellar/beir#beers-course-material-on-ir)
- [Examples and Tutorials](https://github.com/beir-cellar/beir#beers-examples-and-tutorials)
- [Quick Example](https://github.com/beir-cellar/beir#beers-quick-example)
- [Datasets](https://github.com/beir-cellar/beir#beers-download-a-preprocessed-dataset)
    - [Download a preprocessed dataset](https://github.com/beir-cellar/beir#beers-download-a-preprocessed-dataset)
    - [Available Datasets](https://github.com/beir-cellar/beir#beers-available-datasets)
    - [Multilingual Datasets](https://github.com/beir-cellar/beir#beers-multilingual-datasets)
- [Models](https://github.com/beir-cellar/beir#beers-evaluate-a-model)
    - [Evaluate a model](https://github.com/beir-cellar/beir#beers-evaluate-a-model)
    - [Available Models](https://github.com/beir-cellar/beir#beers-available-models)
    - [Evaluate your own Model](https://github.com/beir-cellar/beir#evaluate-your-own-model)
- [Available Metrics](https://github.com/beir-cellar/beir#beers-available-metrics)
- [Citing & Authors](https://github.com/beir-cellar/beir#beers-citing--authors)
- [Collaboration](https://github.com/beir-cellar/beir#beers-collaboration)
- [Contributors](https://github.com/beir-cellar/beir#beers-contributors)


## :beers: Installation

Install via pip:

```python
pip install beir
```

If you want to build from source, use:

```python
$ git clone https://github.com/benchmarkir/beir.git
$ cd beir
$ pip install -e .
```

Tested with python versions 3.6 and 3.7

## :beers: Features 

- Preprocess your own IR dataset or use one of the already-preprocessed 17 benchmark datasets
- Wide settings included, covers diverse benchmarks useful for both academia and industry
- Includes well-known retrieval architectures (lexical, dense, sparse and reranking-based)
- Add and evaluate your own model in a easy framework using different state-of-the-art evaluation metrics

## :beers: Leaderboard

Find below Google Sheets for BEIR Leaderboard. Unfortunately with Markdown the tables were not easy to read.

|           Leaderboard       |      Link     |
| -------------------------   |  ------------ |
|  Dense Retrieval            | [Google Sheet](https://docs.google.com/spreadsheets/d/1L8aACyPaXrL8iEelJLGqlMqXKPX2oSP_R10pZoy77Ns/edit#gid=0) |
| BM25 top-100 + CE Reranking | [Google Sheet](https://docs.google.com/spreadsheets/d/1L8aACyPaXrL8iEelJLGqlMqXKPX2oSP_R10pZoy77Ns/edit#gid=867044147) |

## :beers: Course Material on IR

If you are new to Information Retrieval and wish to understand and learn more about classical or neural IR, we suggest you to look at the open-sourced courses below. 

|      Course          |  University   |  Instructor            |   Link   |  Available |
| -----------------    |  ------------ |  --------------------- | ---------|  --------- |
| Training SOTA Neural Search Models |  Hugging Face |  Nils Reimers          | [Link](https://www.youtube.com/watch?v=XHY-3FzaLGc) | Video |
| BEIR: Benchmarking IR|  UKP Lab      |  Nandan Thakur         | [Link](https://www.youtube.com/watch?v=e9nNr4ugNAo&ab_channel=deepset) | Video + Slides |
| Intro to Advanced IR |  TU Wien'21   |  Sebastian Hofstaetter | [Link](https://github.com/sebastian-hofstaetter/teaching) | Videos + Slides |
| CS224U NLU + IR      |  Stanford'21  |  Omar Khattab          | [Link](http://web.stanford.edu/class/cs224u/) | Slides |
| Pretrained Transformers for Text Ranking: BERT and Beyond    |  MPI, Waterloo'21  |  Andrew Yates, Rodrigo Nogueira, Jimmy Lin | [Link](https://arxiv.org/abs/2010.06467) | PDF |
| BoF Session on IR | NAACL'21 | Sean MacAvaney, Luca Soldaini | [Link](https://docs.google.com/presentation/d/1BkGGnlLLtU-THbqbD60NDZPcPVRSeKxZSy73DKX1u-M/edit) |  Slides |


## :beers: Examples and Tutorials

To easily understand and get your hands dirty with BEIR, we invite you to try our tutorials out :rocket: :rocket:

### :beers: Google Colab

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| How to evaluate pre-trained models on BEIR datasets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing) |

### :beers: Lexical Retrieval (Evaluation)

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| BM25 Retrieval with Elasticsearch  | [evaluate_bm25.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_bm25.py) |
| Anserini-BM25 (Pyserini) Retrieval with Docker  | [evaluate_anserini_bm25.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_anserini_bm25.py) |
| Multilingual BM25 Retrieval with Elasticsearch :new: | [evaluate_multilingual_bm25.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_multilingual_bm25.py) |

### :beers: Dense Retrieval (Evaluation)

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| Exact-search retrieval using (dense) Sentence-BERT | [evaluate_sbert.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_sbert.py) |
| Exact-search retrieval using (dense) ANCE | [evaluate_ance.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_ance.py) |
| Exact-search retrieval using (dense) DPR | [evaluate_dpr.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_dpr.py) |
| Exact-search retrieval using (dense) USE-QA | [evaluate_useqa.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_useqa.py) |
| ANN and Exact-search using Faiss :new: | [evaluate_faiss_dense.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_faiss_dense.py) |
| Retrieval using Binary Passage Retriver (BPR) :new: | [evaluate_bpr.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_bpr.py) |
| Dimension Reduction using PCA :new: | [evaluate_dim_reduction.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_dim_reduction.py) |

### :beers: Sparse Retrieval (Evaluation)

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| Hybrid sparse retrieval using SPARTA | [evaluate_sparta.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/sparse/evaluate_sparta.py) |
| Sparse retrieval using docT5query and Pyserini | [evaluate_anserini_docT5query.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/sparse/evaluate_anserini_docT5query.py) |
| Sparse retrieval using docT5query (MultiGPU) and Pyserini :new: | [evaluate_anserini_docT5query_parallel.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/sparse/evaluate_anserini_docT5query_parallel.py) |
| Sparse retrieval using DeepCT and Pyserini :new: | [evaluate_deepct.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/sparse/evaluate_deepct.py) |

### :beers: Reranking (Evaluation)

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| Reranking top-100 BM25 results with SBERT CE | [evaluate_bm25_ce_reranking.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/reranking/evaluate_bm25_ce_reranking.py) |
| Reranking top-100 BM25 results with Dense Retriever | [evaluate_bm25_sbert_reranking.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/reranking/evaluate_bm25_sbert_reranking.py) |

### :beers: Dense Retrieval (Training)

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| Train SBERT with Inbatch negatives| [train_sbert.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_sbert.py) |
| Train SBERT with BM25 hard negatives| [train_sbert_BM25_hardnegs.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_sbert_BM25_hardnegs.py) |
| Train MSMARCO SBERT with BM25 Negatives | [train_msmarco_v2.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v2.py) |
| Train (SOTA) MSMARCO SBERT with Mined Hard Negatives :new: | [train_msmarco_v3.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3.py) |
| Train (SOTA) MSMARCO BPR with Mined Hard Negatives :new: | [train_msmarco_v3_bpr.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3_bpr.py) |
| Train (SOTA) MSMARCO SBERT with Mined Hard Negatives (Margin-MSE) :new: | [train_msmarco_v3_margin_MSE.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3_margin_MSE.py) |

### :beers: Question Generation

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| Synthetic Query Generation using T5-model | [query_gen.py](https://github.com/beir-cellar/beir/blob/main/examples/generation/query_gen.py) |
| (GenQ) Synthetic QG using T5-model + fine-tuning SBERT | [query_gen_and_train.py](https://github.com/beir-cellar/beir/blob/main/examples/generation/query_gen_and_train.py) |
| Synthetic Query Generation using Multiple GPU and T5 :new: | [query_gen_multi_gpu.py](https://github.com/beir-cellar/beir/blob/main/examples/generation/query_gen_multi_gpu.py) |

### :beers: Benchmarking (Evaluation)

|                          Name                |     Link     |
| -------------------------------------------  |  ----------  |
| Benchmark BM25 (Inference speed) | [benchmark_bm25.py](https://github.com/beir-cellar/beir/blob/main/examples/benchmarking/benchmark_bm25.py) |
| Benchmark Cross-Encoder Reranking (Inference speed) | [benchmark_bm25_ce_reranking.py](https://github.com/beir-cellar/beir/blob/main/examples/benchmarking/benchmark_bm25_ce_reranking.py) |
| Benchmark Dense Retriever (Inference speed) | [benchmark_sbert.py](https://github.com/beir-cellar/beir/blob/main/examples/benchmarking/benchmark_sbert.py) |

## :beers: Quick Example

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

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"), batch_size=16, score_function="cos_sim") # or "dot" for dot-product
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```

## :beers: Download a preprocessed dataset

To load one of the already preprocessed datasets in your current directory as follows:

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
```

This will download the ``scifact`` dataset under the ``datasets`` directory.

For other datasets, just use one of the datasets names, mention below.

## :beers: Available Datasets

Command to generate md5hash using Terminal:  ``md5hash filename.zip``.

| Dataset   | Website| BEIR-Name | Type | Queries  | Corpus | Rel D/Q | Down-load | md5 |
| -------- | -----| ---------| --------- | ----------- | ---------| ---------| :----------: | :------:|
| MSMARCO    | [Homepage](https://microsoft.github.io/msmarco/)| ``msmarco`` | ``train``<br>``dev``<br>``test``|  6,980   |  8.84M     |    1.1 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip) | ``444067daf65d982533ea17ebd59501e4`` |
| MSMARCO v2 | [Homepage](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html)| ``msmarco-v2`` | ``train``<br>``dev1``<br>``dev2``|  4,552<br>4,702   |  138M    |   | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco-v2.zip) | ``ba6238b403f0b345683885cc9390fff5`` |
| TREC-COVID |  [Homepage](https://ir.nist.gov/covidSubmit/index.html)| ``trec-covid``| ``test``| 50|  171K| 493.5 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip) | ``ce62140cb23feb9becf6270d0d1fe6d1`` |
| NFCorpus   | [Homepage](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) | ``nfcorpus`` | ``train``<br>``dev``<br>``test``|  323     |  3.6K     |  38.2 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) | ``a89dba18a62ef92f7d323ec890a0d38d`` |
| BioASQ     | [Homepage](http://bioasq.org) | ``bioasq``|  ``train``<br>``test`` | 500    |  14.91M    |  8.05 | No | [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#2-bioasq) |
| NQ         | [Homepage](https://ai.google.com/research/NaturalQuestions) | ``nq``| ``train``<br>``test``| 3,452   |  2.68M  |  1.2 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip) | ``d4d3d2e48787a744b6f6e691ff534307`` |
| HotpotQA   | [Homepage](https://hotpotqa.github.io) | ``hotpotqa``| ``train``<br>``dev``<br>``test``|  7,405   |  5.23M  |  2.0 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip)  | ``f412724f78b0d91183a0e86805e16114`` |
| FiQA-2018  | [Homepage](https://sites.google.com/view/fiqa/) | ``fiqa`` | ``train``<br>``dev``<br>``test``|  648     |  57K    |  2.6 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip)  | ``17918ed23cd04fb15047f73e6c3bd9d9`` |
| Signal-1M(RT) | [Homepage](https://research.signal-ai.com/datasets/signal1m-tweetir.html)| ``signal1m`` | ``test``| 97   |  2.86M  |  19.6 | No | [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#4-signal-1m) |
| TREC-NEWS  | [Homepage](https://trec.nist.gov/data/news2019.html) | ``trec-news``    | ``test``| 57    |  595K    |  19.6 | No | [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#1-trec-news) |
| ArguAna    | [Homepage](http://argumentation.bplaced.net/arguana/data) | ``arguana``| ``test`` | 1,406     |  8.67K    |  1.0 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip)  | ``8ad3e3c2a5867cdced806d6503f29b99`` |
| Touche-2020| [Homepage](https://webis.de/events/touche-20/shared-task-1.html) | ``webis-touche2020``| ``test``| 49     |  382K    |  19.0 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip) | ``46f650ba5a527fc69e0a6521c5a23563`` |
| CQADupstack| [Homepage](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | ``cqadupstack``| ``test``| 13,145 |  457K  |  1.4 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip) | ``4e41456d7df8ee7760a7f866133bda78`` |
| Quora| [Homepage](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) | ``quora``| ``dev``<br>``test``| 10,000     |  523K    |  1.6 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip) | ``18fb154900ba42a600f84b839c173167`` |
| DBPedia | [Homepage](https://github.com/iai-group/DBpedia-Entity/) | ``dbpedia-entity``| ``dev``<br>``test``| 400    |  4.63M    |  38.2 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip) | ``c2a39eb420a3164af735795df012ac2c`` |
| SCIDOCS| [Homepage](https://allenai.org/data/scidocs) | ``scidocs``| ``test``| 1,000     |  25K    |  4.9 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip) | ``38121350fc3a4d2f48850f6aff52e4a9`` |
| FEVER | [Homepage](http://fever.ai) | ``fever``| ``train``<br>``dev``<br>``test``|  6,666     |  5.42M    |  1.2|  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip)  | ``5a818580227bfb4b35bb6fa46d9b6c03`` |
| Climate-FEVER| [Homepage](http://climatefever.ai) | ``climate-fever``|``test``|  1,535     |  5.42M |  3.0 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip)  | ``8b66f0a9126c521bae2bde127b4dc99d`` |
| SciFact| [Homepage](https://github.com/allenai/scifact) | ``scifact``| ``train``<br>``test``|  300     |  5K    |  1.1 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip)  | ``5f7d1de60b170fc8027bb7898e2efca1`` |
| Robust04 | [Homepage](https://trec.nist.gov/data/robust/04.guidelines.html) | ``robust04``| ``test``| 249  |  528K  |  69.9 |  No  |  [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#3-robust04)  |

## :beers: Multilingual Datasets

| Language | Dataset   | Website| BEIR-Name | Type | Queries  | Corpus | Rel D/Q | Down-load | md5 |
| -------- |  -------- | -----| ---------| -------- | ----------- | ---------| ---------| :----------: | :------:|
|  German  | GermanQuAD | [Homepage](https://deepset.ai/germanquad) | ``germanquad``| ``test``| 2,044  |  2.80M  |  1.0 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/germanquad.zip)  | ``95a581c3162d10915a418609bcce851b`` |
|  Arabic    | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/arabic``| ``train``<br>``dev``<br>``test``| 1,081  | 2.1M  |  1.2 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|  Bengali   | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/bengali``| ``train``<br>``dev``<br>``test``| 111  |  304K  |  1.2 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|  Finnish   | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/finnish``| ``train``<br>``dev``<br>``test``| 1,254 |  1.9M  |  1.2 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|Indonesian  | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/indonesian``| ``train``<br>``dev``<br>``test``| 829  |  1.47M  |  1.2 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
| Japanese   | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/japanese``| ``train``<br>``dev``<br>``test``| 720  |  7M  |  1.3 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|  Korean    | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/korean`` | ``train``<br>``dev``<br>``test``| 421  |  1.5M  |  1.2 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|  Russian   | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/russian``| ``train``<br>``dev``<br>``test``| 995  |  9.6M  |  1.2 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|  Swahili  | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/swahili``| ``train``<br>``dev``<br>``test``| 670  |  136K  |  1.1 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|  Telugu  | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/telugu``| ``train``<br>``dev``<br>``test``| 646  |  548K  |  1.0 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |
|  Thai  | Mr.TyDI | [Homepage](https://github.com/castorini/mr.tydi) | ``mrtydi/thai``| ``train``<br>``dev``<br>``test``| 1,190  | 568K  |  1.1 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mrtydi.zip)  | ``17072d0e1610bd8461d962b8ac560fc5`` |

## :beers: Translated (Multilingual) Datasets

| Language | Dataset   | Website| BEIR-Name | Type | Queries  | Corpus | Rel D/Q | Down-load | md5 |
| -------- |  -------- | -----| ---------| -------- | ----------- | ---------| ---------| :----------: | :------:|
| Spanish  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/spanish``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |
| French  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/french``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |
| Portuguese  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/portuguese``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |
| Italian  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/italian``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |
| Indonesian  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/indonesian``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |
| German  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/german``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |
| Russian  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/russian``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |
| Chinese  | mMARCO    | [Homepage](https://github.com/unicamp-dl/mMARCO) | ``mmarco/chinese``| ``train``<br>``dev`` | 6,980 |  8.84M  |  1.1   |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/mmarco.zip)  | ``b727dbec65315a76bceaff56ad77d2c7`` |


Otherwise, you can load a custom preprocessed dataset in the following way:

```python
from beir.datasets.data_loader import GenericDataLoader

corpus_path = "your_corpus_file.jsonl"
query_path = "your_query_file.jsonl"
qrels_path = "your_qrels_file.tsv"

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()
```

**Make sure that the dataset is in the following format**:
- corpus file: a .jsonl file (jsonlines) that contains a list of dictionaries, each with three fields ``_id`` with unique document identifier, ``title`` with document title (optional) and ``text`` with document paragraph or passage. For example: ``{"_id": "doc1", "title": "Albert Einstein", "text": "Albert Einstein was a German-born...."}``
- queries file: a .jsonl file (jsonlines) that contains a list of dictionaries, each with two fields ``_id`` with unique query identifier and ``text`` with query text. For example: ``{"_id": "q1", "text": "Who developed the mass-energy equivalence formula?"}``
- qrels file: a .tsv file (tab-seperated) that contains three columns, i.e. the query-id, corpus-id and score in this order. Keep 1st row as header. For example: ``q1    doc1    1``


You can also **skip** the dataset loading part and provide directly corpus, queries and qrels in the following way:

```python
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

queries = {
    "q1" : "Who developed the mass-energy equivalence formula?",
    "q2" : "Which beer is brewed with a large proportion of wheat?"
}

qrels = {
    "q1" : {"doc1": 1},
    "q2" : {"doc2": 1},
}

```

### Disclaimer

Similar to Tensorflow [datasets](https://github.com/tensorflow/datasets) or HuggingFace's [datasets](https://github.com/huggingface/datasets) library, we just downloaded and prepared public datasets. We only distribute these datasets in a specific format, but we do not vouch for their quality or fairness, or claim that you have license to use the dataset. It remains the user's responsibility to determine whether you as a user have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this library, feel free to post an issue here or make a pull request!

If you're a dataset owner and wish to include your dataset or model in this library, feel free to post an issue here or make a pull request!


## :beers: Evaluate a model

We include different retrieval architectures and evaluate them all in a zero-shot setup.

### Lexical Retrieval Evaluation using BM25 (Elasticsearch)

```python
from beir.retrieval.search.lexical import BM25Search as BM25

hostname = "your-hostname" #localhost
index_name = "your-index-name" # scifact
initialize = True # True, will delete existing index with same name and reindex all documents
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
```

### Sparse Retrieval using SPARTA

```python
from beir.retrieval.search.sparse import SparseSearch
from beir.retrieval import models

model_path = "BeIR/sparta-msmarco-distilbert-base-v1"
sparse_model = SparseSearch(models.SPARTA(model_path), batch_size=128)
```

### Dense Retrieval using SBERT, ANCE, USE-QA or DPR

```python
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"), batch_size=16, score_function="cos_sim") # or "dot" for dot-product
retriever = EvaluateRetrieval(model)
```

### Reranking using Cross-Encoder model

```python
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
reranker = Rerank(cross_encoder_model, batch_size=128)

# Rerank top-100 results retrieved by BM25
rerank_results = reranker.rerank(corpus, queries, bm25_results, top_k=100)
```

## :beers: Available Models

|  Name     |  Implementation  |
|  -------  |   -------------  |
|  BM25  (Robertson and Zaragoza, 2009) | [https://www.elastic.co/](https://www.elastic.co/) |
| Anserini (Yang et al., 2017) | [https://github.com/castorini/anserini](https://github.com/castorini/anserini) |
|  SBERT (Reimers and Gurevych, 2019)   | [https://www.sbert.net/](https://www.sbert.net/) |
|  ANCE (Xiong et al., 2020) | [https://github.com/microsoft/ANCE](https://github.com/microsoft/ANCE) |
|  DPR (Karpukhin et al., 2020) | [https://github.com/facebookresearch/DPR](https://github.com/facebookresearch/DPR) |
|  USE-QA (Yang et al., 2020) | [https://tfhub.dev/google/universal-sentence-encoder-qa/3](https://tfhub.dev/google/universal-sentence-encoder-qa/3) |
|  SPARTA (Zhao et al., 2020) | [https://huggingface.co/BeIR](https://huggingface.co/BeIR) |
|  ColBERT (Khattab and Zaharia, 2020) | [https://github.com/stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT) |

### Disclaimer

If you use any one of the implementations, please make sure to include the correct citation.

If you implemented a model and wish to update any part of it, or do not want the model to be included, feel free to post an issue here or make a pull request! 

If you implemented a model and wish to include your model in this library, feel free to post an issue here or make a pull request. Otherwise, if you want to evaluate the model on your own, see the following section.

## :beers: Evaluate your own Model

### Dense-Retriever Model (Dual-Encoder)

Mention your dual-encoder model in a class and have two functions: 1. ``encode_queries`` and 2. ``encode_corpus``.

```python
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

class YourCustomDEModel:
    def __init__(self, model_path=None, **kwargs)
        self.model = None # ---> HERE Load your custom model
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        pass
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        pass

custom_model = DRES(YourCustomDEModel(model_path="your-custom-model-path"))
```

### Re-ranking-based Model (Cross-Encoder)

Mention your cross-encoder model in a class and have a single function:  ``predict``

```python
from beir.reranking import Rerank

class YourCustomCEModel:
    def __init__(self, model_path=None, **kwargs)
        self.model = None # ---> HERE Load your custom model
    
    # Write your own score function, which takes in query-document text pairs and returns the similarity scores
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:
        pass # return only the list of float scores

reranker = Rerank(YourCustomCEModel(model_path="your-custom-model-path"), batch_size=128)
```

## :beers: Available Metrics

We evaluate our models using [pytrec_eval](https://github.com/cvangysel/pytrec_eval) and in future we can extend to include more retrieval-based metrics:

- NDCG (``NDCG@k``)
- MAP (``MAP@k``)
- Recall (``Recall@k``)
- Precision (``P@k``)

We also include custom-metrics now which can be used for evaluation, please refer here - [evaluate_custom_metrics.py](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/custom/evaluate_custom_metrics.py)

- MRR (``MRR@k``)
- Capped Recall (``R_cap@k``)
- Hole (``Hole@k``): % of top-k docs retrieved unseen by annotators
- Top-K Accuracy (``Accuracy@k``): % of relevant docs present in top-k results

## :beers: Citing & Authors

If you find this repository helpful, feel free to cite our publication [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663):

```
@inproceedings{
    thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}
```

The main contributors of this repository are:
- [Nandan Thakur](https://github.com/Nthakur20), Personal Website: [nandan-thakur.com](https://nandan-thakur.com)

Contact person: Nandan Thakur, [nandant@gmail.com](mailto:nandant@gmail.com)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## :beers: Collaboration

The BEIR Benchmark has been made possible due to a collaborative effort of the following universities and organizations:
- [UKP Lab, Technical University of Darmstadt](http://www.ukp.tu-darmstadt.de/)
- [University of Waterloo](https://uwaterloo.ca/)
- [HuggingFace](https://huggingface.co/)

## :beers: Contributors

Thanks go to all these wonderful collaborations for their contribution towards the BEIR benchmark:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://www.nandan-thakur.com"><img src="https://avatars.githubusercontent.com/u/30648040?v=4" width="100px;" alt=""/><br /><sub><b>Nandan Thakur</b></sub></a></td>
    <td align="center"><a href="https://www.nils-reimers.de/"><img src="https://avatars.githubusercontent.com/u/10706961?v=4" width="100px;" alt=""/><br /><sub><b>Nils Reimers</b></sub></a></td>
    <td align="center"><a href="https://www.informatik.tu-darmstadt.de/ukp/ukp_home/head_ukp/index.en.jsp"><img src="https://www.informatik.tu-darmstadt.de/media/ukp/pictures_1/people_1/Gurevych_Iryna_500x750_415x415.jpg" width="100px;" alt=""/><br /><sub><b>Iryna Gurevych</b></sub></a></td>
    <td align="center"><a href="https://cs.uwaterloo.ca/~jimmylin/"><img src="https://avatars.githubusercontent.com/u/313837?v=4" width="100px;" alt=""/><br /><sub><b>Jimmy Lin</b></sub></a></td>
    <td align="center"><a href="http://rueckle.net"><img src="https://i1.rgstatic.net/ii/profile.image/601126613295104-1520331161365_Q512/Andreas-Rueckle.jpg" width="100px;" alt=""/><br /><sub><b>Andreas Rücklé</b></sub></a></td>
    <td align="center"><a href="https://www.linkedin.com/in/abhesrivas"><img src="https://avatars.githubusercontent.com/u/19344566?v=4" width="100px;" alt=""/><br /><sub><b>Abhishek Srivastava</b></sub></a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
