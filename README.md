# beir
A heterogeneous benchmark for Information Retrieval

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