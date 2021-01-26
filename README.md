# BeIR: A Heterogeneous Benchmark for Information Retrieval

BeIR provides a benchmark for various diverse IR tasks and a common and easy framework for evaluation of your IR models across a diverse choice of datasets.

The package takes care of the downloading, hosting, preprocessing datasets and providing you in a single easy to understand dataset zip folders. We take care of transforming the dataset and provide 15 diverse datasets used for IR in the both academia and industry, with more to add. Further the package provides an easy framework to evalaute your models against some competitive benchmarks including Sentence-Transformers (SBERT), Dense Passage Retrieval (DPR), Universal Sentence Encoder (USE-QA) and Elastic Search.

### Worried about your dataset not present in the benchmark?

Worry not! You can easily add your dataset into the benchmark by following this data format (here) and you will be able evaluate our models over your dataset.

Want a new dataset? feel free to post an issue here or make a pull request!

### Worried about your model not present in the benchmark?

Worry not! You can also evaluate your own private models using the benchmark, and only required to return a dictionary with mappings (check it out here!) and you can evaluate your model using our code. 

Want us to add a new model? feel free to post an issue here or make a pull request!


## Installation

Install via pip:

```python
pip install beir
```

If you want to build from source, use:

```
$ git clone https://github.com/beir-nlp/beir.git
$ pip install -e .
```

Tested with python versions 3.6 and 3.7

## Getting Started

First download and unzip a dataset.

```python
from beir import util

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
out_path = "datasets/trec-covid.zip"
out_dir = "datasets"

util.download_url(url, out_path)
util.unzip(out_path, out_dir)
```

Then load the dataset using our Generic Data Loader.

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

Now you can use either Sentence-transformers, DPR or USE-QA as your dense retriever model.
The format of results is identical to that of qrels.

```python
from beir.retrieval.evaluation import EvaluateRetrieval

retriever = EvaluateRetrieval(model="sbert", model_name="distilroberta-base-msmarco-v2")
# retriever = EvaluateRetrieval(model="dpr")
# retriever = EvaluateRetrieval(model="use-qa")

results = retriever.retrieve(corpus, queries, qrels)

for query_id, metadata in results.items():
    for doc_id, retrieval_score in metadata.items():
        print(query_id, doc_id, retrieval_score)
# 1     005b2j4b    0.77
# 1     00fmeepz    0.6
# ...
```

Finally, after retrieving the scores, you can evaluate them using qrels and results files.
We evaluate using NDCG, MAP, Recall and Precision scores @ K.
We advise to chose NDCG@10, explanation mentioned in our upcoming paper.

```python
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

for key, value in ndcg.items():
    print(key, value) 
# ndcg@1    0.3456
# ndcg@3    0.4567
# ...
```

## Datasets available in the benchmark

1. Download datasets using ``datasets/download_data.py``
2. Evaluate using ``evaluate_model.py`` wherein set line 4 => ``data_path = "../datasets/{dataset-name}"``