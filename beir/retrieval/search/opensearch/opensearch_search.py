from __future__ import annotations

import logging
import time

from opensearchpy import OpenSearch
from opensearchpy.helpers import streaming_bulk

logger = logging.getLogger("OpenSearchEngine")
#tracer.setLevel(logging.CRITICAL)  # suppressing INFO messages for opensearch


# We named it OpenSearchEngine to avoid conflict with OpenSearch class from the client library
class OpenSearchEngine:
    def __init__(self, os_credentials: dict[str, object]):
        logging.info("Activating OpenSearch....")
        logging.info("OpenSearch Credentials: %s", os_credentials)
        self.index_name = os_credentials["index_name"]
        self.check_index_name()

        # Same language analyzers as Elasticsearch
        self.languages = [
            "arabic", "armenian", "basque", "bengali", "brazilian", "bulgarian",
            "catalan", "cjk", "czech", "danish", "dutch", "english", "estonian",
            "finnish", "french", "galician", "german", "greek", "hindi",
            "hungarian", "indonesian", "irish", "italian", "latvian", "lithuanian",
            "norwegian", "persian", "portuguese", "romanian", "russian", "sorani",
            "spanish", "swedish", "turkish", "thai"
        ]

        self.language = os_credentials["language"]
        self.check_language_supported()

        self.text_key = os_credentials["keys"]["body"]
        self.embedding_key = os_credentials["keys"]["embedding"]
        self.title_key = os_credentials["keys"]["title"]
        self.number_of_shards = os_credentials["number_of_shards"]

        # OpenSearch client initialization
        self.os = OpenSearch(
            hosts=[os_credentials["hostname"]],
            timeout=os_credentials["timeout"],
            retry_on_timeout=os_credentials["retry_on_timeout"],
            max_retries=3,
            use_ssl=os_credentials.get("use_ssl", False),
            verify_certs=os_credentials.get("verify_certs", False),
            ssl_show_warn=os_credentials.get("ssl_show_warn", False),
            http_auth=os_credentials.get("http_auth", None)
        )

    def check_language_supported(self):
        if self.language.lower() not in self.languages:
            raise ValueError(
                f"Invalid Language: {self.language}, not supported by OpenSearch. Languages Supported: {self.languages}"
            )

    def check_index_name(self):
        for char in r'#:\/*?"<>|,':
            if char in self.index_name:
                raise ValueError(r'Invalid OpenSearch Index, must not contain the characters ===> #:\/*?"<>|,')

        if self.index_name.startswith(("_", "-", "+")):
            raise ValueError("Invalid OpenSearch Index, must not start with characters ===> _ or - or +")

        if self.index_name in [".", ".."]:
            raise ValueError("Invalid OpenSearch Index, must not be . or ..")

        if not self.index_name.islower():
            raise ValueError("Invalid OpenSearch Index, must be lowercase")

    def create_index(self):
        logging.info(f"Creating fresh OpenSearch-Index named - {self.index_name}")

        try:
            mapping = {
                "settings": {
                    "number_of_shards": self.number_of_shards if self.number_of_shards != "default" else 1,
                    "analysis": {
                        "analyzer": {
                            "default": {"type": self.language}
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        self.title_key: {"type": "text", "analyzer": self.language},
                        self.text_key: {"type": "text", "analyzer": self.language}
                    }
                }
            }

            self.os.indices.create(
                index=self.index_name,
                body=mapping,
                ignore=[400]  # 400: IndexAlreadyExistsException
            )
        except Exception as e:
            logging.error(f"Unable to create Index in OpenSearch. Reason: {e}")

    def delete_index(self):
        logging.info(f"Deleting previous OpenSearch-Index named - {self.index_name}")
        try:
            self.os.indices.delete(
                index=self.index_name,
                ignore=[400, 404]  # 404: IndexDoesntExistException
            )
        except Exception as e:
            logging.error(f"Unable to delete Index in OpenSearch. Reason: {e}")

    def bulk_add_to_index(self, generate_actions, progress):
        for ok, action in streaming_bulk(
            client=self.os,
            index=self.index_name,
            actions=generate_actions,
        ):
            progress.update(1)
        progress.reset()
        progress.close()

    def lexical_search(self, text: str, top_hits: int, ids: list[str] = None, skip: int = 0) -> dict[str, object]:
        req_body = {
            "query": {
                "multi_match": {
                    "query": text,
                    "type": "best_fields",
                    "fields": [self.text_key, self.title_key],
                    "tie_breaker": 0.5
                }
            }
        }

        if ids:
            req_body = {
                "query": {
                    "bool": {
                        "must": req_body["query"],
                        "filter": {"ids": {"values": ids}}
                    }
                }
            }

        res = self.os.search(
            index=self.index_name,
            body=req_body,
            size=skip + top_hits,
            search_type="dfs_query_then_fetch"
        )

        hits = [(hit["_id"], hit["_score"]) for hit in res["hits"]["hits"][skip:]]
        return self.hit_template(os_res=res, hits=hits)

    def lexical_multisearch(self, texts: list[str], top_hits: int, skip: int = 0) -> list[dict[str, object]]:
        assert skip + top_hits <= 10000, "OpenSearch Window too large, Max-Size = 10000"
        
        request = []
        for text in texts:
            req_head = {"index": self.index_name, "search_type": "dfs_query_then_fetch"}
            req_body = {
                "_source": False,
                "query": {
                    "multi_match": {
                        "query": text,
                        "type": "best_fields",
                        "fields": [self.title_key, self.text_key],
                        "tie_breaker": 0.5
                    }
                },
                "size": skip + top_hits
            }
            request.extend([req_head, req_body])

        res = self.os.msearch(body=request)
        
        result = []
        for resp in res["responses"]:
            hits = [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"][skip:]]
            result.append(self.hit_template(os_res=resp, hits=hits))
        return result

    def generate_actions(self, dictionary: dict[str, dict[str, str]], update: bool = False):
        for _id, value in dictionary.items():
            if not update:
                doc = {
                    "_id": str(_id),
                    "_op_type": "index",
                    "refresh": "wait_for",
                    self.text_key: value[self.text_key],
                    self.title_key: value[self.title_key]
                }
            else:
                doc = {
                    "_id": str(_id),
                    "_op_type": "update",
                    "refresh": "wait_for",
                    "doc": {
                        self.text_key: value[self.text_key],
                        self.title_key: value[self.title_key]
                    }
                }
            yield doc

    def hit_template(self, os_res: dict[str, object], hits: list[tuple[str, float]]) -> dict[str, object]:
        return {
            "meta": {
                "total": os_res["hits"]["total"]["value"],
                "took": os_res["took"],
                "num_hits": len(hits)
            },
            "hits": hits
        }

    def configure_ml_settings(self):
        """Configure OpenSearch ML Commons settings for text embedding.
        
        Sets required cluster settings for running ML models:
        - Allows running ML on any node
        - Enables model access control
        - Sets native memory threshold
        """
        logging.info("Configuring ML cluster settings...")
        try:
            settings = {
                "persistent": {
                    "plugins.ml_commons.only_run_on_ml_node": "false",
                    "plugins.ml_commons.model_access_control_enabled": "true", 
                    "plugins.ml_commons.native_memory_threshold": "99"
                }
            }
            self.os.cluster.put_settings(body=settings)
        except Exception as e:
            logging.error(f"Unable to configure ML cluster settings. Reason: {e}")

    def register_model_group(self, name: str = "local_model_group", description: str = "A model group for local models"):
        """Register a new model group for ML models.
        
        Args:
            name: Name of the model group
            description: Description of the model group's purpose
        """
        logging.info(f"Registering model group: {name}")
        try:
            body = {
                "name": name,
                "description": description
            }
            
            response = self.os.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/model_groups/_register",
                body=body
            )
            return response
        except Exception as e:
            logging.error(f"Unable to register model group. Reason: {e}")

    def register_model(
        self, 
        name: str = "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b",
        version: str = "1.0.2",
        model_group_id: str = None,
        model_format: str = "TORCH_SCRIPT"
    ) -> dict:
        """Register a new ML model.
        
        Args:
            name: Name/path of the model
            version: Version of the model
            model_group_id: ID of the model group to associate with (from register_model_group response)
            model_format: Format of the model (e.g., TORCH_SCRIPT)
        
        Returns:
            dict: Response from the API containing task_id and status
        """
        if not model_group_id:
            raise ValueError("model_group_id is required. Get it from register_model_group() response.")
        
        logging.info(f"Registering model: {name}")
        try:
            body = {
                "name": name,
                "version": version,
                "model_group_id": model_group_id,
                "model_format": model_format
            }
            
            response = self.os.transport.perform_request(
                method="POST",
                url="/_plugins/_ml/models/_register",
                body=body
            )
            return response
        except Exception as e:
            logging.error(f"Unable to register model. Reason: {e}")
            raise

    def wait_for_model_registration(self, task_id: str, timeout: int = 300) -> str:
        """Wait for the model registration to complete.

        Args:
            task_id: ID of the task (from registration response)
            timeout: Timeout in seconds

        Returns:
            bool: model_id if the model is registered, throw exception otherwise
        """
        logging.info(f"Waiting for model registration to complete: {task_id}")
        try:
            response = self.os.transport.perform_request(
                method="GET",
                url=f"/_plugins/_ml/tasks/{task_id}"
            )
            start_time = time.time()
            while response["state"] != "COMPLETED":
                time.sleep(1)
                response = self.os.transport.perform_request(
                    method="GET",
                    url=f"/_plugins/_ml/tasks/{task_id}"
                )
                if time.time() - start_time > timeout:
                    raise TimeoutError("Model registration timed out")
            return response["model_id"]
        except Exception as e:
            logging.error(f"Unable to wait for model registration. Reason: {e}")
            raise

    def deploy_model(self, model_id: str) -> dict:
        """Deploy a model to the cluster.

        Args:
            model_id: ID of the model (from register_model response)

        Returns:
            dict: Response from the API containing task_id and status
        """
        logging.info(f"Deploying model: {model_id}")
        try:
            response = self.os.transport.perform_request(
                method="POST",
                url=f"/_plugins/_ml/models/{model_id}/_deploy",
                body={}
            )
            return response
        except Exception as e:
            logging.error(f"Unable to deploy model. Reason: {e}")
            raise

    def wait_for_model_deployment(self, task_id: str, timeout: int = 300) -> str:
        """Wait for the model deployment to complete.

        Args:
            task_id: ID of the task (from deploy_model response)
            timeout: Timeout in seconds

        Returns:
            bool: model_id if the model is deployed, throw exception otherwise
        """
        logging.info(f"Waiting for model deployment to complete: {task_id}")
        try:
            response = self.os.transport.perform_request(
                method="GET",
                url=f"/_plugins/_ml/tasks/{task_id}"
            )
            start_time = time.time()
            while response["state"] != "COMPLETED":
                time.sleep(1)
                response = self.os.transport.perform_request(
                    method="GET",
                    url=f"/_plugins/_ml/tasks/{task_id}"
                )
                if time.time() - start_time > timeout:
                    raise TimeoutError("Model deployment timed out")
            return response["model_id"]
        except Exception as e:
            logging.error(f"Unable to wait for model deployment. Reason: {e}")
            raise

    def create_ingest_pipeline(self, model_id: str):
        """Create an ingest pipeline for text embedding.

        Args:
            model_id: ID of the model (from register_model response)
        """
        logging.info("Creating Ingest Pipeline for Neural Search")
        try:
            pipeline_config = {
                "description": "An NLP ingest pipeline",
                "processors": [
                    {
                        "text_embedding": {
                            "model_id": model_id,
                            "field_map": {
                                self.text_key: self.embedding_key
                            }
                        }
                    }
                ]
            }
            self.os.ingest.put_pipeline(
                id="nlp-ingest-pipeline",
                body=pipeline_config
            )
        except Exception as e:
            logging.error(f"Unable to create Ingest Pipeline in OpenSearch. Reason: {e}")

    def create_neural_search_index(self, dimension: int = 768):
        """
        Create neural search index with knn_vector field
        """
        logging.info(f"Creating fresh OpenSearch-Index named - {self.index_name}")
        try:
            mapping = {
                "settings": {
                    "number_of_shards": self.number_of_shards if self.number_of_shards != "default" else 1,
                    "analysis": {
                        "analyzer": {
                            "default": {"type": self.language}
                        }
                    },
                    "index.knn": True,
                    "default_pipeline": "nlp-ingest-pipeline"
                },
                "mappings": {
                    "properties": {
                        self.title_key: {"type": "text", "analyzer": self.language},
                        self.text_key: {"type": "text", "analyzer": self.language},
                        self.embedding_key: {"type": "knn_vector", "dimension": dimension,
                            "method": {
                                "name": "disk_ann",
                                "space_type": "l2",
                                "engine": "jvector"
                            }
                        }
                    }
                }
            }

            self.os.indices.create(
                index=self.index_name,
                body=mapping
            )
        except Exception as e:
            logging.error(f"Unable to create Index in OpenSearch. Reason: {e}")

    def neural_search(
        self, 
        text: str, 
        model_id: str,
        top_hits: int = 5,
        exclude_embeddings: bool = True
    ) -> dict[str, object]:
        """Perform neural search using text embeddings.
        
        Args:
            text: Query text to search for
            model_id: ID of the model to use for embedding
            top_hits: Number of top results to return (default: 5)
            exclude_embeddings: Whether to exclude embeddings from response (default: True)
        
        Returns:
            dict: Search results containing hits and metadata
        """
        logging.info(f"Performing neural search for: {text}")
        try:
            body = {
                "_source": {
                    "excludes": [self.embedding_key] if exclude_embeddings else []
                },
                "query": {
                    "neural": {
                        self.embedding_key: {
                            "query_text": text,
                            "model_id": model_id,
                            "k": top_hits
                        }
                    }
                }
            }

            response = self.os.search(
                index=self.index_name,
                body=body
            )
            
            hits = [(hit["_id"], hit["_score"]) for hit in response["hits"]["hits"]]
            return self.hit_template(os_res=response, hits=hits)
        
        except Exception as e:
            logging.error(f"Neural search failed. Reason: {e}")
            raise

    def neural_multisearch(
        self,
        texts: list[str],
        model_id: str,
        top_hits: int = 5,
        exclude_embeddings: bool = True
    ) -> list[dict[str, object]]:
        """Perform neural search for multiple queries in batch.
        
        Args:
            texts: List of query texts to search for
            model_id: ID of the model to use for embedding
            top_hits: Number of top results to return per query
            exclude_embeddings: Whether to exclude embeddings from response
        
        Returns:
            list: List of search results, one per query
        """
        logging.info(f"Performing batch neural search for {len(texts)} queries")
        try:
            request = []
            for text in texts:
                req_head = {"index": self.index_name}
                req_body = {
                    "_source": {
                        "excludes": [self.embedding_key] if exclude_embeddings else []
                    },
                    "query": {
                        "neural": {
                            self.embedding_key: {
                                "query_text": text,
                                "model_id": model_id,
                                "k": top_hits
                            }
                        }
                    }
                }
                request.extend([req_head, req_body])

            response = self.os.msearch(body=request)
            logger.info(f"Batch neural search response: {response}")
            results = []
            for resp in response["responses"]:
                hits = [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]
                results.append(self.hit_template(os_res=resp, hits=hits))
            return results
        
        except Exception as e:
            logging.error(f"Batch neural search failed. Reason: {e}")
            raise

    def delete_ingest_pipeline(self):
        """Delete the ingest pipeline for text embedding."""
        logging.info("Deleting Ingest Pipeline for Neural Search")
        try:
            self.os.ingest.delete_pipeline(
                id="nlp-ingest-pipeline"
            )
        except Exception as e:
            logging.error(f"Unable to delete Ingest Pipeline in OpenSearch. Reason: {e}")

    def undeploy_model(self, model_id: str):
        """Undeploy a model from the cluster.

        Args:
            model_id: ID of the model to undeploy
        """
        logging.info(f"Undeploying model: {model_id}")
        try:
            self.os.transport.perform_request(
                method="POST",
                url=f"/_plugins/_ml/models/{model_id}/_undeploy"
            )
        except Exception as e:
            logging.error(f"Unable to undeploy model. Reason: {e}")

    def delete_model(self, model_id: str):
        """Delete a model from the cluster.

        Args:
            model_id: ID of the model to delete
        """
        logging.info(f"Deleting model: {model_id}")
        try:
            self.os.transport.perform_request(
                method="DELETE",
                url=f"/_plugins/_ml/models/{model_id}"
            )
        except Exception as e:
            logging.error(f"Unable to delete model. Reason: {e}")
