# BeIR: A Heterogeneous Benchmark for Information Retrieval

BeIR provides a benchmark for various diverse IR tasks, and provides a common and easy tool, for you to evaluate your IR models over the diverse benchmark.

The package takes care of the downloading, hosting, preprocessing datasets and providing you in a single easy to understand zip folders. We take care of transforming the dataset publicly online and provide datasets for over 15 diverse IR tasks, with more to add.

Further the package provides an easy interface for you to evalaute your IR models across these diverse datasets, to evaluate your IR models against some competitive benchmark models including Sentence-Transformers (SBERT), Dense Passage Retrieval (DPR), Universal Sentence Encoder (USE-QA) and Elastic-Search.

## Worried about your dataset not present in the benchmark?

Worry not! You can easily add your dataset into the benchmark by following this data format (here) and you will be able evaluate our models over your dataset.
Want a new dataset? feel free to post an issue or make a pull request!

## Worried about your model not present in the benchmark?

Worry not! You can also evaluate your own private models using the benchmark, and only required to return a dictionary with mappings (check it out here!) and you can evaluate your model using our code. 
Want us to add a new model? feel free to post an issue or make a pull request!


## Installation

Install via pip:

```
pip install beir
```

If you want to build from source, use:

```
$ git clone https://github.com/beir-nlp/beir.git
$ pip install -e .
```

Tested with python versions 3.6 and 3.7
## Steps To Follow

1. Download datasets using ``datasets/download_data.py``
2. Evaluate using ``evaluate_model.py`` wherein set line 4 => ``data_path = "../datasets/{dataset-name}"``