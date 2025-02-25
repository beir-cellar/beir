import logging
import os
import pathlib
import random
import time

from beir import util
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Just some code to print debug information to stdout
logging.basicConfig(level=logging.INFO)
#### /print debug information to stdout


# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == "__main__":
    dataset = "fiqa"

    #### Download fiqa.zip dataset and unzip the dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data path where fiqa has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) fiqa/corpus.jsonl  (format: jsonlines)
    # (2) fiqa/queries.jsonl (format: jsonlines)
    # (3) fiqa/qrels/test.tsv (format: tsv ("\t"))

    #### Load our locally downloaded datasets via HFDataLoader to save RAM (i.e. do not load the whole corpus in RAM)
    corpus, queries, qrels = HFDataLoader(data_folder=data_path, streaming=False).load(split="test")

    #### You can use our custom hosted BEIR datasets on HuggingFace again to save RAM (streaming=True) ####
    # corpus, queries, qrels = HFDataLoader(hf_repo=f"BeIR/{dataset}", streaming=False, keep_in_memory=False).load(split="test")

    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    #### Provide any pretrained sentence-transformers model
    #### The model was fine-tuned using cosine-similarity.
    #### Complete list - https://www.sbert.net/docs/pretrained_models.html
    beir_model = models.SentenceBERT(
        "NovaSearch/stella_en_400M_v5",
        max_length=512,
        prompt_names={"query": "s2p_query", "passage": None},
        trust_remote_code=True,
    )

    #### Start with Parallel search and evaluation
    model = DRES(beir_model, batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time.time()
    results = retriever.retrieve(corpus, queries)
    end_time = time.time()
    print(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    ### If you want to save your results and runfile (useful for reranking)
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    #### Save the evaluation runfile & results
    util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
    util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)

    #### Print top-k documents retrieved ####
    top_k = 10

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    query = queries.filter(lambda x: x["id"] == query_id)[0]["text"]
    logging.info(f"Query : {query}\n")

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        doc = corpus.filter(lambda x: x["id"] == doc_id)[0]
        # Format: Rank x: ID [Title] Body
        logging.info(f"Rank {rank + 1}: {doc_id} [{doc.get('title')}] - {doc.get('text')}\n")
