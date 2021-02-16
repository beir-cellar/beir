# Deep Dive into Evaluation of Retrieval Models

## 1. Data Downloading and Loading

First download and unzip a dataset. Load the dataset with our data loader.

Click here to view [**15+ Datasets**](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/) available in BEIR.

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
out_dir = "datasets"
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where trec-covid has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
```

## 2. Model Loading

Now, you can use either Sentence-transformers, DPR or USE-QA as your dense retriever model.

```python
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Load SBERT model ####
model = DRES(models.SentenceBERT("distilroberta-base-msmarco-v2"))

#### Load DPR model ####
# model = DRES(models.DPR(
#     'facebook/dpr-question_encoder-single-nq-base',
#     'facebook/dpr-ctx_encoder-single-nq-base' ))

#### Load USE-QA model ####
# model = DRES(models.UseQA("https://tfhub.dev/google/universal-sentence-encoder-qa/3"))
```

Or if you wish to use lexical retrieval, we provide support with Elasticsearch.
```python
from beir.retrieval.search.lexical import BM25Search as BM25

#### Provide parameters for elastic-search
hostname = "your-es-hostname-here" # localhost for default
index_name = "your-index-name-here"
model = BM25(index_name=index_name, hostname=hostname)
```
## 3. Retriever Search and Evaluation

Format of ``results`` is identical to that of ``qrels``. You can evaluate your IR performance using ``qrels`` and ``results``.
We find ``NDCG@10`` score for all datasets, for more details on why check our upcoming paper.

```python
from beir.retrieval.evaluation import EvaluateRetrieval

retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```