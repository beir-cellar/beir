from __future__ import annotations

import logging

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

tracer = logging.getLogger("elasticsearch")
tracer.setLevel(logging.CRITICAL)  # supressing INFO messages for elastic-search


class ElasticSearch:
    def __init__(self, es_credentials: dict[str, object]):
        logging.info("Activating Elasticsearch....")
        logging.info("Elastic Search Credentials: %s", es_credentials)
        self.index_name = es_credentials["index_name"]
        self.check_index_name()

        # https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html
        self.languages = [
            "arabic",
            "armenian",
            "basque",
            "bengali",
            "brazilian",
            "bulgarian",
            "catalan",
            "cjk",
            "czech",
            "danish",
            "dutch",
            "english",
            "estonian",
            "finnish",
            "french",
            "galician",
            "german",
            "greek",
            "hindi",
            "hungarian",
            "indonesian",
            "irish",
            "italian",
            "latvian",
            "lithuanian",
            "norwegian",
            "persian",
            "portuguese",
            "romanian",
            "russian",
            "sorani",
            "spanish",
            "swedish",
            "turkish",
            "thai",
        ]

        self.language = es_credentials["language"]
        self.check_language_supported()

        self.text_key = es_credentials["keys"]["body"]
        self.title_key = es_credentials["keys"]["title"]
        self.number_of_shards = es_credentials["number_of_shards"]

        self.es = Elasticsearch(
            [es_credentials["hostname"]],
            timeout=es_credentials["timeout"],
            retry_on_timeout=es_credentials["retry_on_timeout"],
            maxsize=es_credentials["maxsize"],
        )

    def check_language_supported(self):
        """Check Language Supported in Elasticsearch"""
        if self.language.lower() not in self.languages:
            raise ValueError(
                f"Invalid Language: {self.language}, not supported by Elasticsearch. Languages Supported: \
            {self.languages}"
            )

    def check_index_name(self):
        """Check Elasticsearch Index Name"""
        # https://stackoverflow.com/questions/41585392/what-are-the-rules-for-index-names-in-elastic-search
        # Check 1: Must not contain the characters ===> #:\/*?"<>|,
        for char in r'#:\/*?"<>|,':
            if char in self.index_name:
                raise ValueError(r'Invalid Elasticsearch Index, must not contain the characters ===> #:\/*?"<>|,')

        # Check 2: Must not start with characters ===> _-+
        if self.index_name.startswith(("_", "-", "+")):
            raise ValueError("Invalid Elasticsearch Index, must not start with characters ===> _ or - or +")

        # Check 3: must not be . or ..
        if self.index_name in [".", ".."]:
            raise ValueError("Invalid Elasticsearch Index, must not be . or ..")

        # Check 4: must be lowercase
        if not self.index_name.islower():
            raise ValueError("Invalid Elasticsearch Index, must be lowercase")

    def create_index(self):
        """Create Elasticsearch Index"""
        logging.info(f"Creating fresh Elasticsearch-Index named - {self.index_name}")

        try:
            if self.number_of_shards == "default":
                mapping = {
                    "mappings": {
                        "properties": {
                            self.title_key: {"type": "text", "analyzer": self.language},
                            self.text_key: {"type": "text", "analyzer": self.language},
                        }
                    }
                }
            else:
                mapping = {
                    "settings": {"number_of_shards": self.number_of_shards},
                    "mappings": {
                        "properties": {
                            self.title_key: {"type": "text", "analyzer": self.language},
                            self.text_key: {"type": "text", "analyzer": self.language},
                        }
                    },
                }

            self.es.indices.create(
                index=self.index_name, body=mapping, ignore=[400]
            )  # 400: IndexAlreadyExistsException
        except Exception as e:
            logging.error(f"Unable to create Index in Elastic Search. Reason: {e}")

    def delete_index(self):
        """Delete Elasticsearch Index"""

        logging.info(f"Deleting previous Elasticsearch-Index named - {self.index_name}")
        try:
            self.es.indices.delete(index=self.index_name, ignore=[400, 404])  # 404: IndexDoesntExistException
        except Exception as e:
            logging.error(f"Unable to create Index in Elastic Search. Reason: {e}")

    def bulk_add_to_index(self, generate_actions, progress):
        """Bulk indexing to elastic search using generator actions

        Args:
            generate_actions (generator function): generator function must be provided
            progress (tqdm.tqdm): tqdm progress_bar
        """
        for ok, action in streaming_bulk(
            client=self.es,
            index=self.index_name,
            actions=generate_actions,
        ):
            progress.update(1)
        progress.reset()
        progress.close()

    def lexical_search(self, text: str, top_hits: int, ids: list[str] = None, skip: int = 0) -> dict[str, object]:
        """[summary]

        Args:
            text (str): query text
            top_hits (int): top k hits to retrieved
            ids (list[str], optional): Filter results for only specific ids. Defaults to None.

        Returns:
            dict[str, object]: Hit results
        """
        req_body = {
            "query": {
                "multi_match": {
                    "query": text,
                    "type": "best_fields",
                    "fields": [self.text_key, self.title_key],
                    "tie_breaker": 0.5,
                }
            }
        }

        if ids:
            req_body = {
                "query": {
                    "bool": {
                        "must": req_body["query"],
                        "filter": {"ids": {"values": ids}},
                    }
                }
            }

        res = self.es.search(
            search_type="dfs_query_then_fetch",
            index=self.index_name,
            body=req_body,
            size=skip + top_hits,
        )

        hits = []

        for hit in res["hits"]["hits"][skip:]:
            hits.append((hit["_id"], hit["_score"]))

        return self.hit_template(es_res=res, hits=hits)

    def lexical_multisearch(self, texts: list[str], top_hits: int, skip: int = 0) -> dict[str, object]:
        """Multiple Query search in Elasticsearch

        Args:
            texts (list[str]): Multiple query texts
            top_hits (int): top k hits to be retrieved
            skip (int, optional): top hits to be skipped. Defaults to 0.

        Returns:
            dict[str, object]: Hit results
        """
        request = []

        assert skip + top_hits <= 10000, "Elastic-Search Window too large, Max-Size = 10000"

        for text in texts:
            req_head = {"index": self.index_name, "search_type": "dfs_query_then_fetch"}
            req_body = {
                "_source": False,  # No need to return source objects
                "query": {
                    "multi_match": {
                        "query": text,  # matching query with both text and title fields
                        "type": "best_fields",
                        "fields": [self.title_key, self.text_key],
                        "tie_breaker": 0.5,
                    }
                },
                "size": skip + top_hits,  # The same paragraph will occur in results
            }

            request.extend([req_head, req_body])

        res = self.es.msearch(body=request)

        result = []
        for resp in res["responses"]:
            responses = resp["hits"]["hits"][skip:]

            hits = []
            for hit in responses:
                hits.append((hit["_id"], hit["_score"]))

            result.append(self.hit_template(es_res=resp, hits=hits))
        return result

    def generate_actions(self, dictionary: dict[str, dict[str, str]], update: bool = False):
        """Iterator function for efficient addition to Elasticsearch
        Ref: https://stackoverflow.com/questions/35182403/bulk-update-with-pythons-elasticsearch
        """
        for _id, value in dictionary.items():
            if not update:
                doc = {
                    "_id": str(_id),
                    "_op_type": "index",
                    "refresh": "wait_for",
                    self.text_key: value[self.text_key],
                    self.title_key: value[self.title_key],
                }
            else:
                doc = {
                    "_id": str(_id),
                    "_op_type": "update",
                    "refresh": "wait_for",
                    "doc": {
                        self.text_key: value[self.text_key],
                        self.title_key: value[self.title_key],
                    },
                }

            yield doc

    def hit_template(self, es_res: dict[str, object], hits: list[tuple[str, float]]) -> dict[str, object]:
        """Hit output results template

        Args:
            es_res (dict[str, object]): Elasticsearch response
            hits (list[tuple[str, float]]): Hits from Elasticsearch

        Returns:
            dict[str, object]: Hit results
        """
        result = {
            "meta": {
                "total": es_res["hits"]["total"]["value"],
                "took": es_res["took"],
                "num_hits": len(hits),
            },
            "hits": hits,
        }
        return result
