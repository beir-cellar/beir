<h1 align="center">
<img style="vertical-align:middle" width="450" height="180" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" />
</h1>

<p align="center">
    <a href="https://github.com/beir-cellar/beir/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/beir-cellar/beir.svg">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/pypi/pyversions/beir?logo=pypi&style=flat&color=blue">
    </a>
    <a href="https://github.com/beir-cellar/beir/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/beir-cellar/beir?logo=github&style=flat&color=green">
    </a>
    <a href="https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://pepy.tech/project/beir">
        <img alt="Downloads" src="https://img.shields.io/pypi/dm/beir?logo=pypi&style=flat&color=orange">
    </a>
    <a href="https://github.com/beir-cellar/beir/">
        <img alt="Open Source" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://openreview.net/forum?id=wCu6T5xFjeJ">Paper</a> |
        <a href="#beers-installation">Installation</a> |
        <a href="#beers-quick-example">Quick Example</a> |
        <a href="#beers-available-datasets">Datasets</a> |
        <a href="https://github.com/beir-cellar/beir/wiki">Wiki</a> |
        <a href="https://huggingface.co/BeIR">Hugging Face</a>
    <p>
</h4>

<!-- > The development of BEIR benchmark is supported by: -->

<h3 align="center">
    <a href="http://www.ukp.tu-darmstadt.de"><img style="float: left; padding: 2px 7px 2px 7px;" width="220" height="100" src="./images/ukp.png" /></a>
    <a href="https://www.tu-darmstadt.de/"><img style="float: middle; padding: 2px 7px 2px 7px;" width="250" height="90" src="./images/tu-darmstadt.png" /></a>
    <a href="https://uwaterloo.ca"><img style="float: right; padding: 2px 7px 2px 7px;" width="320" height="100" src="./images/uwaterloo.png" /></a>
</h3>

<h3 align="center">
    <a href="https://huggingface.co/"><img style="float: middle; padding: 2px 7px 2px 7px;" width="400" height="80" src="./images/HF.png" /></a>
</h3>

## :beers: What is it?

**BEIR** is a **heterogeneous benchmark** containing diverse IR tasks. It also provides a **common and easy framework** for evaluation of your NLP-based retrieval models within the benchmark.

For **an overview**, checkout our **new wiki** page: [https://github.com/beir-cellar/beir/wiki](https://github.com/beir-cellar/beir/wiki).

For **models and datasets**, checkout out **Hugging Face (HF)** page: [https://huggingface.co/BeIR](https://huggingface.co/BeIR).

For more information, checkout out our publications:

- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://openreview.net/forum?id=wCu6T5xFjeJ) (NeurIPS 2021, Datasets and Benchmarks Track)
- [Resources for Brewing BEIR: Reproducible Reference Models and an Official Leaderboard](https://dl.acm.org/doi/10.1145/3626772.3657862) (SIGIR 2024 Resource Track)

## :beers: Installation

Install via pip:

```python
pip install beir
```

If you want to build from source, use:

```python
$ git clone https://github.com/beir-cellar/beir.git
$ cd beir
$ pip install -e .
```

Tested with python versions 3.9+

## :beers: Features

- Preprocess your own IR dataset or use one of the already-preprocessed 17 benchmark datasets
- Wide settings included, covers diverse benchmarks useful for both academia and industry
- Evaluates well-known retrieval architectures (lexical, dense, sparse and reranking-based)
- Add and evaluate your own model in a easy framework using different state-of-the-art evaluation metrics

## :beers: Quick Example

For other example codes, please refer to our **[Examples and Tutorials](https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials)** Wiki page.

<details><summary><b>Quick Example with Sentence-BERT</b></summary>

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
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(models.SentenceBERT("Alibaba-NLP/gte-modernbert-base"), batch_size=16)

retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" for dot product
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
```
</details>

<details><summary><b>Quick Example with LoRA & vLLM</b></summary>
First install peft, vllm & accelerate using the following installations:

```
pip install peft
pip install accelerate
pip install vllm
```

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
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### You can also merge the LoRA model weights into the original base model for faster inference.
#### Checkout: https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/dense/evaluate_lora_vllm.py

#### Load the vLLM embed model and retrieve using cosine-similarity
model = DRES(
    models.VLLMEmbed(
        model_path="Qwen/Qwen2.5-7B",
        lora_name_or_path="rlhn/Qwen2.5-7B-rlhn-400K",
        max_length=512,
        lora_r=16,
        pooling="eos",
        append_eos_token=True,
        normalize=True,
        prompts={"query": "query: ", "passage": "passage: "},
        convert_to_numpy=True
    ),
    batch_size=128,
)

retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" for dot product
results = retriever.encode_and_retrieve(corpus, queries, encode_output_path="./qwen_embeddings/")

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
```
</details>

<details><summary><b>Quick Example with HuggingFace</b></summary>
if you use `encode_and_retrieve()` make sure you install faiss with `pip install faiss-cpu`.

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
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Load the Huggingface model and retrieve using cosine-similarity
query_prompt = "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: "

model = DRES(
    models.HuggingFace(
        model_path="intfloat/e5-mistral-7b-instruct",
        max_length=512,
        pooling="eos",
        append_eos_token=True,
        normalize=True,
        prompts={"query": query_prompt, "passage": ""},
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16"
    ),
    batch_size=128,
)

retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" for dot product
results = retriever.encode_and_retrieve(corpus, queries, encode_output_path="./embeddings/")

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
```
</details>

<details><summary><b>Quick Example with APIs, e.g. Cohere</b></summary>

Install Cohere API using `pip install cohere` & if you are using `encode_and_retrieve()` install faiss with `pip install faiss-cpu`.
```python
from beir import util, LoggingHandler
from beir.retrieval import apis
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
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

cohere_api_key = os.getenv("COHERE_API_KEY")
#### Load the Cohere API Embed model and retrieve using cosine-similarity
model = DRES(
    apis.CohereEmbedAPI(
        api_key=cohere_api_key, 
        model_path="embed-v4.0", 
        normalize=True, 
        torch_dtype="float32"
    ),
    batch_size=96,
)

retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" for dot product
results = retriever.encode_and_retrieve(corpus, queries, encode_output_path="./cohere/embeddings/")

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
```
</details>

## :beers: Available Datasets

Command to generate md5hash using Terminal:  ``md5sum filename.zip``.

You can view all datasets available **[here](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/)** or on **[Hugging Face](https://huggingface.co/BeIR)**.


| Dataset   | Website| BEIR-Name | Public? | Type | Queries  | Corpus | Rel D/Q | Down-load | md5 |
| -------- | -----| ---------| ------- | --------- | ----------- | ---------| ---------| :----------: | :------:|
| MSMARCO    | [Homepage](https://microsoft.github.io/msmarco/)| ``msmarco`` | ✅ | ``train``<br>``dev``<br>``test``|  6,980   |  8.84M     |    1.1 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip) | ``444067daf65d982533ea17ebd59501e4`` |
| TREC-COVID |  [Homepage](https://ir.nist.gov/covidSubmit/index.html)| ``trec-covid``| ✅ | ``test``| 50|  171K| 493.5 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip) | ``ce62140cb23feb9becf6270d0d1fe6d1`` |
| NFCorpus   | [Homepage](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) | ``nfcorpus`` | ✅ |``train``<br>``dev``<br>``test``|  323     |  3.6K     |  38.2 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip) | ``a89dba18a62ef92f7d323ec890a0d38d`` |
| BioASQ     | [Homepage](http://bioasq.org) | ``bioasq``| ❌ | ``train``<br>``test`` | 500 |  14.91M    |  4.7 | No | [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#2-bioasq) |
| NQ         | [Homepage](https://ai.google.com/research/NaturalQuestions) | ``nq``| ✅ | ``train``<br>``test``| 3,452   |  2.68M  |  1.2 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip) | ``d4d3d2e48787a744b6f6e691ff534307`` |
| HotpotQA   | [Homepage](https://hotpotqa.github.io) | ``hotpotqa``| ✅ |``train``<br>``dev``<br>``test``|  7,405   |  5.23M  |  2.0 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip)  | ``f412724f78b0d91183a0e86805e16114`` |
| FiQA-2018  | [Homepage](https://sites.google.com/view/fiqa/) | ``fiqa`` | ✅ | ``train``<br>``dev``<br>``test``|  648     |  57K    |  2.6 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip)  | ``17918ed23cd04fb15047f73e6c3bd9d9`` |
| Signal-1M(RT) | [Homepage](https://research.signal-ai.com/datasets/signal1m-tweetir.html)| ``signal1m`` | ❌ | ``test``| 97   |  2.86M  |  19.6 | No | [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#4-signal-1m) |
| TREC-NEWS  | [Homepage](https://trec.nist.gov/data/news2019.html) | ``trec-news`` | ❌ | ``test``| 57    |  595K    |  19.6 | No | [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#1-trec-news) |
| Robust04 | [Homepage](https://trec.nist.gov/data/robust/04.guidelines.html) | ``robust04``| ❌ | ``test``| 249  |  528K  |  69.9 |  No  |  [How to Reproduce?](https://github.com/beir-cellar/beir/blob/main/examples/dataset#3-robust04)  |
| ArguAna    | [Homepage](http://argumentation.bplaced.net/arguana/data) | ``arguana``| ✅ |``test`` | 1,406     |  8.67K    |  1.0 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip)  | ``8ad3e3c2a5867cdced806d6503f29b99`` |
| Touche-2020| [Homepage](https://webis.de/events/touche-20/shared-task-1.html) | ``webis-touche2020``| ✅ | ``test``| 49     |  382K    |  19.0 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip) | ``46f650ba5a527fc69e0a6521c5a23563`` |
| CQADupstack| [Homepage](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) | ``cqadupstack``| ✅ | ``test``| 13,145 |  457K  |  1.4 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip) | ``4e41456d7df8ee7760a7f866133bda78`` |
| Quora| [Homepage](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) | ``quora``| ✅ | ``dev``<br>``test``| 10,000     |  523K    |  1.6 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip) | ``18fb154900ba42a600f84b839c173167`` |
| DBPedia | [Homepage](https://github.com/iai-group/DBpedia-Entity/) | ``dbpedia-entity``| ✅ | ``dev``<br>``test``| 400    |  4.63M    |  38.2 | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip) | ``c2a39eb420a3164af735795df012ac2c`` |
| SCIDOCS| [Homepage](https://allenai.org/data/scidocs) | ``scidocs``| ✅ | ``test``| 1,000     |  25K    |  4.9 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip) | ``38121350fc3a4d2f48850f6aff52e4a9`` |
| FEVER | [Homepage](http://fever.ai) | ``fever``| ✅ | ``train``<br>``dev``<br>``test``|  6,666     |  5.42M    |  1.2|  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip)  | ``5a818580227bfb4b35bb6fa46d9b6c03`` |
| Climate-FEVER| [Homepage](http://climatefever.ai) | ``climate-fever``| ✅ |``test``|  1,535     |  5.42M |  3.0 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip)  | ``8b66f0a9126c521bae2bde127b4dc99d`` |
| SciFact| [Homepage](https://github.com/allenai/scifact) | ``scifact``| ✅ | ``train``<br>``test``|  300     |  5K    |  1.1 |  [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip)  | ``5f7d1de60b170fc8027bb7898e2efca1`` |


## :beers: Additional Information

We also provide a variety of additional information in our **[Wiki](https://github.com/beir-cellar/beir/wiki)** page.
Please refer to these pages for the following:


### Quick Start

- [Installing BEIR](https://github.com/beir-cellar/beir/wiki/Installing-beir)
- [Examples and Tutorials](https://github.com/beir-cellar/beir/wiki/Examples-and-tutorials)

### Datasets

- [Datasets Available](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [Multilingual Datasets](https://github.com/beir-cellar/beir/wiki/Multilingual-datasets)
- [Load your Custom Dataset](https://github.com/beir-cellar/beir/wiki/Load-your-custom-dataset)

### Models
- [Models Available](https://github.com/beir-cellar/beir/wiki/Models-available)
- [Evaluate your Custom Model](https://github.com/beir-cellar/beir/wiki/Evaluate-your-custom-model)

### Metrics

- [Metrics Available](https://github.com/beir-cellar/beir/wiki/Metrics-available)

### Miscellaneous

- [BEIR Leaderboard](https://github.com/beir-cellar/beir/wiki/Leaderboard)
- [Couse Material on IR](https://github.com/beir-cellar/beir/wiki/Course-material-on-ir)

## :beers: Disclaimer

Similar to Tensorflow [datasets](https://github.com/tensorflow/datasets) or Hugging Face's [datasets](https://github.com/huggingface/datasets) library, we just downloaded and prepared public datasets. We only distribute these datasets in a specific format, but we do not vouch for their quality or fairness, or claim that you have license to use the dataset. It remains the user's responsibility to determine whether you as a user have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.

If you're a dataset owner and wish to update any part of it, or do not want your dataset to be included in this library, feel free to post an issue here or make a pull request!

If you're a dataset owner and wish to include your dataset or model in this library, feel free to post an issue here or make a pull request!

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

If you use any baseline score from the BEIR leaderboard, feel free to cite our publication [Resources for Brewing BEIR: Reproducible Reference Models and an Official Leaderboard](https://arxiv.org/abs/2306.07471)
```
@inproceedings{kamalloo:2024,
    author = {Kamalloo, Ehsan and Thakur, Nandan and Lassance, Carlos and Ma, Xueguang and Yang, Jheng-Hong and Lin, Jimmy},
    title = {Resources for Brewing BEIR: Reproducible Reference Models and Statistical Analyses},
    year = {2024},
    isbn = {9798400704314},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3626772.3657862},
    doi = {10.1145/3626772.3657862},
    abstract = {BEIR is a benchmark dataset originally designed for zero-shot evaluation of retrieval models across 18 different domain/task combinations. In recent years, we have witnessed the growing popularity of models based on representation learning, which naturally begs the question: How effective are these models when presented with queries and documents that differ from the training data? While BEIR was designed to answer this question, our work addresses two shortcomings that prevent the benchmark from achieving its full potential: First, the sophistication of modern neural methods and the complexity of current software infrastructure create barriers to entry for newcomers. To this end, we provide reproducible reference implementations that cover learned dense and sparse models. Second, comparisons on BEIR are performed by reducing scores from heterogeneous datasets into a single average that is difficult to interpret. To remedy this, we present meta-analyses focusing on effect sizes across datasets that are able to accurately quantify model differences. By addressing both shortcomings, our work facilitates future explorations in a range of interesting research questions.},
    booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {1431–1440},
    numpages = {10},
    keywords = {domain generalization, evaluation, reproducibility},
    location = {Washington DC, USA},
    series = {SIGIR '24}
}
```

The main contributors of this repository are:
- [Nandan Thakur](https://github.com/Nthakur20), Personal Website: [thakur-nandan.gitub.io](https://thakur-nandan.github.io)

Contact person: Nandan Thakur, [nandant@gmail.com](mailto:nandant@gmail.com)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## :beers: Collaboration

The BEIR Benchmark has been made possible due to a collaborative effort of the following universities and organizations:
- [UKP Lab, Technical University of Darmstadt](http://www.ukp.tu-darmstadt.de/)
- [University of Waterloo](https://uwaterloo.ca/)
- [Hugging Face](https://huggingface.co/)

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
