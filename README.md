# beir
A heterogeneous benchmark for Information Retrieval

## Installation
We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v3.1.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

**Install with pip**

```
pip install -U beir
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/beir-nlp/beir) and install it directly from the source code:
````
pip install -e .
```` 
## Steps To Follow

1. Download datasets using ``datasets/download_data.py``
2. Evaluate using ``evaluate_model.py`` wherein set line 4 => ``data_path = "../datasets/{dataset-name}"``