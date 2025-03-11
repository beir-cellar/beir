from __future__ import annotations

import time

import tqdm
import logging

from ...base import BaseSearch
from ..opensearch_search import OpenSearchEngine
logger = logging.getLogger("NeuralSearch")

def sleep(seconds):
    if seconds:
        time.sleep(seconds)


class HybridSearch(BaseSearch):
    def __init__(
        self,
        index_name: str,
        hostname: str = "localhost",
        keys: dict[str, str] = {"title": "title", "body": "txt", "embedding": "embedding"},
        language: str = "english",
        batch_size: int = 128,
        timeout: int = 100,
        retry_on_timeout: bool = True,
        maxsize: int = 24,
        number_of_shards: int = "default",
        initialize: bool = True,
        sleep_for: int = 2
    ):
        self.model_id = None
        self.results = {}
        self.batch_size = batch_size
        self.initialize = initialize
        self.sleep_for = sleep_for
        self.config = {
            "hostname": hostname,
            "index_name": index_name,
            "keys": keys,
            "timeout": timeout,
            "retry_on_timeout": retry_on_timeout,
            "maxsize": maxsize,
            "number_of_shards": number_of_shards,
            "language": language,
        }
        # Initialize OpenSearch engine
        self.os_engine = OpenSearchEngine(self.config)
        if self.initialize:
            self.initialise()

    def initialise(self):
        """
        Initialise OpenSearch for neural search.
        """
        # Setup ML infrastructure
        self.os_engine.configure_ml_settings()
        # Register model group and get ID
        model_group_response = self.os_engine.register_model_group()
        model_group_id = model_group_response["model_group_id"]
        # Register model using group ID
        model_register_response = self.os_engine.register_model(model_group_id=model_group_id)
        logger.info(f"Model registration response: {model_register_response}")
        self.model_id = self.os_engine.wait_for_model_deployment(task_id=model_register_response["task_id"]) # Use this ID in create_ingest_pipeline
        logger.info(f"Model ID: {self.model_id}")
        deploy_task_response = self.os_engine.deploy_model(self.model_id)
        logger.info(f"Model deployment response: {deploy_task_response}")
        self.os_engine.wait_for_model_deployment(task_id=deploy_task_response["task_id"])
        # Create ingest pipeline
        self.os_engine.create_ingest_pipeline(model_id=self.model_id)
        # Create index
        self.os_engine.create_neural_search_index()
        # Create search pipeline
        self.os_engine.create_search_pipeline()

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        # Index the corpus within elastic-search
        # False, if the corpus has been already indexed
        if self.initialize:
            self.index(corpus)
            # Sleep for few seconds so that elastic-search indexes the docs properly
            sleep(self.sleep_for)

        # retrieve neural search results from OpenSearch
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in query_ids]

        for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc="que"):
            query_ids_batch = query_ids[start_idx : start_idx + self.batch_size]
            results = self.os_engine.hybrid_multisearch(
                texts=queries[start_idx : start_idx + self.batch_size],
                model_id=self.model_id,
                top_hits=top_k + 1,
            )  # Add 1 extra if query is present with documents

            for query_id, hit in zip(query_ids_batch, results):
                scores = {}
                for corpus_id, score in hit["hits"]:
                    if corpus_id != query_id:  # query doesnt return in results
                        scores[corpus_id] = score
                    self.results[query_id] = scores

        return self.results

    def index(self, corpus: dict[str, dict[str, str]]):
        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        # dictionary structure = {_id: {title_key: title, text_key: text}}
        dictionary = {
            idx: {
                self.config["keys"]["title"]: corpus[idx].get("title", None),
                self.config["keys"]["body"]: corpus[idx].get("text", None),
            }
            for idx in list(corpus.keys())
        }
        self.os_engine.bulk_add_to_index(
            generate_actions=self.os_engine.generate_actions(dictionary=dictionary, update=False),
            progress=progress,
        )

    def cleanup(self):
        self.os_engine.delete_search_pipeline()
        self.os_engine.delete_index()
        self.os_engine.delete_ingest_pipeline()
        self.os_engine.undeploy_model(self.model_id)
        self.os_engine.delete_model(self.model_id)
