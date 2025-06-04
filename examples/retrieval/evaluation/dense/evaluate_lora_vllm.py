"""
Allows for evaluating LoRA fine-tuned dense retrievers, i.e., in the tevatron format on BEIR datasets with VLLM.
You would need to install the vllm client library. You can install it using `pip install vllm`.
Also great to check whether you have accelerate installed correctly: `pip install accelerate`.
VLLM encoding is much faster than the HuggingFace implementation.
Example usage: CUDA_VISIBLE_DEVICES=0 python evaluate_lora_vllm.py
"""

import logging
import os
import pathlib
import random
from time import time

from peft import PeftConfig, PeftModel
from transformers import AutoModel, AutoTokenizer

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


def get_model(peft_model_name_or_path: str, cache_dir: str = None) -> tuple[PeftModel, str]:
    config = PeftConfig.from_pretrained(peft_model_name_or_path, cache_dir=cache_dir)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, cache_dir=cache_dir)
    model = PeftModel.from_pretrained(base_model, peft_model_name_or_path, cache_dir=cache_dir)
    model = model.merge_and_unload()
    model.eval()
    return model, config.base_model_name_or_path


def main():
    dataset = "nfcorpus"

    #### Download nfcorpus.zip dataset and unzip the dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) nfcorpus/corpus.jsonl  (format: jsonlines)
    # (2) nfcorpus/queries.jsonl (format: jsonlines)
    # (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #### Dense Retrieval using E5 or Tevatron with Hugging Face ####
    #### Provide any pretrained E5 or Tevatron fine-tuned model
    #### The model was fine-tuned using normalization & cosine-similarity.

    ##################################################################
    #### Faster if you merge the LoRA adapter weights with the base model weights.
    merged_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "merged")
    model, base_model_path = get_model("rlhn/Qwen2.5-7B-rlhn-400K")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model.half()
    os.makedirs(merged_model_path, exist_ok=True)

    # Save the merged model and tokenizer
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    ##################################################################

    ## Parameters
    model_name_or_path = merged_model_path
    max_length = 512
    lora_r = 16
    pooling = "eos"
    normalize = True
    append_eos_token = True
    query_prompt = "query: "
    passage_prompt = "passage: "

    ### if you do not want to merge the model and use the LoRA adapter weights, you can use the following code:
    ### We advise to merge the LoRA adapter weights with the base model weights for faster inference.
    ### However merging may take extra disk space as you are saving the merged model weights locally.
    # model_path = "Qwen/Qwen2.5-7B"
    # lora_model_name_or_path = "rlhn/Qwen2.5-7B-rlhn-400K"
    # rest of the parameters stay the same
    # dense_model = models.VLLMEmbed(
    #     model_path=model_path,
    #     lora_name_or_path=lora_model_name_or_path,
    #     ....)

    #### Configuration for Qwen2.5-7B embedding model
    dense_model = models.VLLMEmbed(
        model_path=model_name_or_path,
        lora_r=lora_r,
        max_length=max_length,
        append_eos_token=append_eos_token,  # add [EOS] token to the end of the input
        pooling=pooling,
        normalize=normalize,
        prompts={"query": query_prompt, "passage": passage_prompt},
        convert_to_numpy=True,
    )

    model = DRES(dense_model, batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = retriever.encode_and_retrieve(corpus, queries, encode_output_path="./embeddings/")
    end_time = time()
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
    logging.info(f"Query : {queries[query_id]}\n")

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")

    #### NDCG@K results should look like this:
    # "NDCG@1": 0.48452,
    # "NDCG@3": 0.44744,
    # "NDCG@5": 0.42539,
    # "NDCG@10": 0.39287,
    # "NDCG@100": 0.36137,
    # "NDCG@1000": 0.4509


if __name__ == "__main__":
    main()
