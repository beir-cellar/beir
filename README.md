
<!-- <h1>
<img style="vertical-align:middle" width="120" height="120" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo.png" />
BeIR: A Heterogeneous Benchmark for IR
</h1> -->

<!-- <h1 text-align= "center">
    <img width="300" height="120" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" style="vertical-align: middle;"/>
</h1> -->

<p>
<img style="vertical-align:middle" width="300" height="120" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" />
</p>

<!-- <h3 align="center">
BEIR: A heterogeneous benchmark for Information Retrieval
</h3> -->


![PyPI](https://img.shields.io/pypi/v/beir)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Nthakur20/StrapDown.js/graphs/commit-activity)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchmarkir/beir/blob/main/examples/retrieval/Retrieval_Example.ipynb)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/benchmarkir/beir/)

## What is it?

**BEIR** consists a **heterogeneous benchmark** for diverse sentence or passage IR level tasks. It also provides a **common and easy framework** for evaluation of your NLP models on them.

The package takes care of the downloading, hosting, preprocessing datasets and providing you in a single easy to understand dataset zip folders. We take care of transforming the dataset and provide 15 diverse datasets used for IR in the both academia and industry, with more to add. Further the package provides an easy framework to evalaute your models against some competitive benchmarks including Sentence-Transformers (SBERT), Dense Passage Retrieval (DPR), Universal Sentence Encoder (USE-QA) and Elastic Search.

### Worried about your dataset or model not present in the benchmark?

Worry not! You can easily add your dataset into the benchmark by following this data format (here) and also you are free to evaluate your own model and required to return a dictionary with mappings (here) and you can evaluate your IR model using our easy plugin code.

Want us to add a new dataset or a new model? feel free to post an issue here or make a pull request!

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

Try it out live with our [Google Colab Example](https://colab.research.google.com/github/benchmarkir/beir/blob/main/examples/retrieval/Retrieval_Example.ipynb).

First download and unzip a dataset. Click here to [**view**](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/) all datasets available.

```python
from beir import util

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
out_path = "datasets/trec-covid.zip"
out_dir = "datasets"

util.download_url(url, out_path)
util.unzip(out_path, out_dir)
```

Then load the dataset using our Generic Data Loader, (Wonderful right)

```python
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

data_path = "datasets/trec-covid/"
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
```

Now, you can use either Sentence-transformers, DPR or USE-QA as your dense retriever model.
Format of ``results`` is identical to that of ``qrels``.

```python
from beir.retrieval.evaluation import EvaluateRetrieval

retriever = EvaluateRetrieval(model="sbert", model_name="distilroberta-base-msmarco-v2") 
# retriever = EvaluateRetrieval(model="dpr")
# retriever = EvaluateRetrieval(model="use-qa")

results = retriever.retrieve(corpus, queries, qrels)
```

Finally after retrieving, you can evaluate your IR performance using ``qrels`` and ``results``.
We find ``NDCG@10`` score for all datasets, for more details on why check our upcoming paper.

```python
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

for key, value in ndcg.items():
    print(key, value) 
# ndcg@1    0.3456
# ndcg@3    0.4567
# ...
```

## Examples

For all examples, see below:

### Retrieval
- [Google Colab Example](https://colab.research.google.com/github/benchmarkir/beir/blob/main/examples/retrieval/Retrieval_Example.ipynb)
- [Exact Search Retrieval using SBERT](https://github.com/benchmarkir/beir/blob/main/examples/retrieval/evaluate_sbert.py)
- [Exact Search Retrieval using DPR](https://github.com/benchmarkir/beir/blob/main/examples/retrieval/evaluate_dpr.py)
- [Exact Search Retrieval using USE-QA](https://github.com/benchmarkir/beir/blob/main/examples/retrieval/evaluate_useqa.py)

## Datasets

Available datasets include:

- TREC-COVID    [[homepage](https://ir.nist.gov/covidSubmit/index.html)]
- NFCorpus      [[homepage](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)]
- NQ            [[homepage](https://ai.google.com/research/NaturalQuestions)]
- HotpotQA      [[homepage](https://hotpotqa.github.io/)]
- NewsQA        [[homepage](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)]
- FiQA          [[homepage](https://sites.google.com/view/fiqa/home)]
- ArguAna       [[homepage](http://argumentation.bplaced.net/arguana/data)]
- Touche-2020   [[homepage](https://webis.de/events/touche-20/)]
- CQaDupstack   [[homepage](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)]
- Quora         [[homepage](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)]
- DBPedia-v2    [[homepage](https://iai-group.github.io/DBpedia-Entity/)]
- SCIDOCS       [[homepage](https://allenai.org/data/scidocs)]
- FEVER         [[homepage](https://fever.ai/)]
- Climate-FEVER [[homepage](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)]
- Signal-1M (Optional) [[homepage](https://research.signal-ai.com/datasets/signal1m-tweetir.html)]
- BioASQ (Optional) [[homepage](http://bioasq.org/)]

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

| Domain     |Dataset       | BM25    | SBERT   | USE-QA  | DPR     |
| :---------:|------------: |:------: |:------: |:------: |:------: |
|            | TREC-COVID   |         |         |         |         |
| Bio-Medical| BioASQ       |         |         |         |         |
|            | NFCorpus     |         |         |         |         |
|            |              |         |         |         |         |
| Question   | NQ           |         |         |         |         |
| Answering  | HotpotQA     |         |         |         |         |
|            |              |         |         |         |         |
| News       | NewsQA       |         |         |         |         |
|            |              |         |         |         |         |
| Twitter    | Signal-1M    |         |         |         |         |
|            |              |         |         |         |         |
| Finance    | FiQA-2018    |         |         |         |         |
| Argument   | ArguAna      |         |         |         |         |
|            | Touche-2020  |         |         |         |         |
|            |              |         |         |         |         |
| Duplicate  | CQaDupstack  |         |         |         |         |
| Question   | Quora        |         |         |         |         |
|            |              |         |         |         |         |
|  Entity    | DBPedia-v2   |         |         |         |         |
|            |              |         |         |         |         |
| Scientific | SCIDOCS      |         |         |         |         |
|            |              |         |         |         |         |
| Claim      | FEVER        |         |         |         |         |
|Verification|Climate-FEVER |         |         |         |         |


## Citing & Authors

The main contributors of this repository are:
- [Nandan Thakur](https://github.com/Nthakur20) 

Contact person: Nandan Thakur, [nandant@gmail.com](mailto:nandant@gmail.com)

[https://www.ukp.tu-darmstadt.de/](https://www.ukp.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

