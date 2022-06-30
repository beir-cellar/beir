from collections import defaultdict
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
import time

import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(level=logging.INFO)
#### /print debug information to stdout


#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == "__main__":

    dataset = "fiqa"

    #### Download fiqa.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
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
    beir_model = models.SentenceBERT("msmarco-distilbert-base-tas-b")

    #### Start with Parallel search and evaluation
    model = DRPES(beir_model, batch_size=512, target_devices=None, corpus_chunk_size=512*2)
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time.time()
    results = retriever.retrieve(corpus, queries)
    end_time = time.time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

    #### Optional: Stop the proccesses in the pool
    # beir_model.doc_model.stop_multi_process_pool(pool)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

    #### Print top-k documents retrieved ####
    top_k = 10

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    query = queries.filter(lambda x: x['id']==query_id)[0]['text']
    logging.info("Query : %s\n" % query)

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        doc = corpus.filter(lambda x: x['id']==doc_id)[0]
        # Format: Rank x: ID [Title] Body
        logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, doc.get("title"), doc.get("text")))