import shutil
import re
from statistics import mean, median
from collections import Counter
from typing import Dict, Optional
from vespa.application import Vespa
from vespa.package import ApplicationPackage, Field, FieldSet, RankProfile, QueryField
from vespa.query import QueryModel, OR, RankProfile as Ranking, WeakAnd
from vespa.deployment import VespaDocker
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError

REPLACE_SYMBOLS = ["(", ")", " -", " +"]
QUOTES = [
    "\u0022",  # quotation mark (")
    "\u0027",  # apostrophe (')
    "\u00ab",  # left-pointing double-angle quotation mark
    "\u00bb",  # right-pointing double-angle quotation mark
    "\u2018",  # left single quotation mark
    "\u2019",  # right single quotation mark
    "\u201a",  # single low-9 quotation mark
    "\u201b",  # single high-reversed-9 quotation mark
    "\u201c",  # left double quotation mark
    "\u201d",  # right double quotation mark
    "\u201e",  # double low-9 quotation mark
    "\u201f",  # double high-reversed-9 quotation mark
    "\u2039",  # single left-pointing angle quotation mark
    "\u203a",  # single right-pointing angle quotation mark
    "\u300c",  # left corner bracket
    "\u300d",  # right corner bracket
    "\u300e",  # left white corner bracket
    "\u300f",  # right white corner bracket
    "\u301d",  # reversed double prime quotation mark
    "\u301e",  # double prime quotation mark
    "\u301f",  # low double prime quotation mark
    "\ufe41",  # presentation form for vertical left corner bracket
    "\ufe42",  # presentation form for vertical right corner bracket
    "\ufe43",  # presentation form for vertical left corner white bracket
    "\ufe44",  # presentation form for vertical right corner white bracket
    "\uff02",  # fullwidth quotation mark
    "\uff07",  # fullwidth apostrophe
    "\uff62",  # halfwidth left corner bracket
    "\uff63",  # halfwidth right corner bracket
]
REPLACE_SYMBOLS.extend(QUOTES)


def replace_symbols(x):
    for symbol in REPLACE_SYMBOLS:
        x = x.replace(symbol, "")
    return x


class VespaLexicalSearch:
    def __init__(
        self,
        application_name: str,
        match_phase: str = "or",
        rank_phase: str = "bm25",
        deployment_parameters: Optional[Dict] = None,
        initialize: bool = True,
    ):
        self.results = {}
        self.application_name = application_name.replace("-", "")
        assert match_phase in [
            "or",
            "weak_and",
        ], "'match_phase' should be either 'or' or 'weak_and'"
        self.match_phase = match_phase
        assert rank_phase in [
            "bm25",
            "native_rank",
        ], "'rank_phase' should be either 'bm25' or 'native_rank'"
        self.rank_phase = rank_phase
        self.deployment_parameters = deployment_parameters
        self.initialize = initialize
        self.vespa_docker = None
        if self.initialize:
            self.app = self.initialise()
            assert (
                self.app.get_application_status().status_code == 200
            ), "Application status different than 200."
        else:
            self.vespa_docker = VespaDocker.from_container_name_or_id(
                self.application_name
            )
            assert self.deployment_parameters is not None, (
                "if 'initialize' is set to false, 'deployment_parameters' should contain Vespa "
                "connection parameters such as 'url' and 'port'"
            )
            self.app = Vespa(**self.deployment_parameters)
            assert (
                self.app.get_application_status().status_code == 200
            ), "Application status different than 200."

    def initialise(self):
        #
        # Create Vespa application package
        #
        app_package = ApplicationPackage(name=self.application_name)
        app_package.schema.add_fields(
            Field(name="id", type="string", indexing=["attribute", "summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index"],
                index="enable-bm25",
            ),
            Field(
                name="body",
                type="string",
                indexing=["index"],
                index="enable-bm25",
            ),
        )
        app_package.schema.add_field_set(
            FieldSet(name="default", fields=["title", "body"])
        )
        app_package.schema.add_rank_profile(
            rank_profile=RankProfile(
                name="bm25", first_phase="bm25(title) + bm25(body)"
            )
        )
        app_package.schema.add_rank_profile(
            rank_profile=RankProfile(
                name="native_rank", first_phase="nativeRank(title,body)"
            )
        )
        app_package.query_profile.add_fields(QueryField(name="maxHits", value=10000))
        #
        # Deploy application
        #
        if not self.deployment_parameters:
            self.deployment_parameters = {"port": 8089, "container_memory": "12G"}
        self.vespa_docker = VespaDocker(**self.deployment_parameters)
        app = self.vespa_docker.deploy(application_package=app_package)
        app.delete_all_docs(
            content_cluster_name=self.application_name + "_content",
            schema=self.application_name,
        )
        return app

    def remove_app(self):
        if self.vespa_docker:
            shutil.rmtree(
                self.application_name, ignore_errors=True
            )  # remove application package folder
            self.vespa_docker.container.stop(timeout=600)  # stop docker container
            self.vespa_docker.container.remove()  # rm docker container

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(10))
    def send_query_batch(
        self, query_batch, query_model, hits, timeout=100, async_connections=50
    ):
        query_results = self.app.query_batch(
            query_batch=query_batch,
            query_model=query_model,
            connections=async_connections,
            total_timeout=timeout * len(query_batch),
            hits=hits,
            **{"timeout": str(timeout) + " s", "ranking.softtimeout.enable": "false"}
        )
        return query_results

    def process_queries(
        self, query_ids, queries, query_model, hits, batch_size, timeout=100, async_connections=50
    ):
        results = {}
        assert len(query_ids) == len(
            queries
        ), "There must be one query_id for each query."
        query_id_batches = [
            query_ids[i : i + batch_size] for i in range(0, len(query_ids), batch_size)
        ]
        query_batches = [
            queries[i : i + batch_size] for i in range(0, len(queries), batch_size)
        ]
        for idx, (query_id_batch, query_batch) in enumerate(
            zip(query_id_batches, query_batches)
        ):
            print(
                "{}, {}, {}: {}/{}".format(
                    self.application_name,
                    self.match_phase,
                    self.rank_phase,
                    idx,
                    len(query_batches),
                )
            )
            try:
                query_results = self.send_query_batch(
                    query_batch=query_batch,
                    query_model=query_model,
                    hits=hits,
                    timeout=timeout,
                    async_connections=async_connections
                )
                number_hits = [x.number_documents_retrieved for x in query_results]
                status_code_summary = Counter([x.status_code for x in query_results])
                print(
                    "Sucessfull queries: {}/{}\nDocuments retrieved. Min: {}, Max: {}, Mean: {}, Median: {}.".format(
                        status_code_summary[200],
                        len(query_batch),
                        min(number_hits),
                        max(number_hits),
                        round(mean(number_hits), 2),
                        round(median(number_hits), 2),
                    )
                )
            except RetryError:
                continue
            for (query_id, query_result) in zip(query_id_batch, query_results):
                scores = {}
                try:
                    if query_result.hits:
                        for hit in query_result.hits:
                            corpus_id = hit["fields"]["id"]
                            if (
                                corpus_id != query_id
                            ):  # See https://github.com/UKPLab/beir/issues/72
                                scores[corpus_id] = hit["relevance"]
                except KeyError:
                    continue
                results[query_id] = scores
        return results

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        *args,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:

        if self.initialize:
            _ = self.index(corpus)

        # retrieve results from BM25
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in query_ids]

        queries = [
            re.sub(" +", " ", replace_symbols(x)).strip() for x in queries
        ]  # remove quotes and double spaces from queries

        if self.match_phase == "or":
            match_phase = OR()
        elif self.match_phase == "weak_and":
            match_phase = WeakAnd(hits=top_k)
        else:
            ValueError("'match_phase' should be either 'or' or 'weak_and'")

        if self.rank_phase not in ["bm25", "native_rank"]:
            ValueError("'rank_phase' should be either 'bm25' or 'native_rank'")

        query_model = QueryModel(
            name=self.match_phase + "_" + self.rank_phase,
            match_phase=match_phase,
            rank_profile=Ranking(name=self.rank_phase, list_features=False),
        )

        self.results = self.process_queries(
            query_ids=query_ids,
            queries=queries,
            query_model=query_model,
            hits=top_k,
            batch_size=1000,
            timeout="100 s",
        )
        return self.results

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(10))
    def send_feed_batch(self, feed_batch, total_timeout=10000):
        feed_results = self.app.feed_batch(
            batch=feed_batch, total_timeout=total_timeout
        )
        return feed_results

    def index(self, corpus: Dict[str, Dict[str, str]], batch_size=1000):
        batch_feed = [
            {
                "id": idx,
                "fields": {
                    "id": idx,
                    "title": corpus[idx].get("title", None),
                    "body": corpus[idx].get("text", None),
                },
            }
            for idx in list(corpus.keys())
        ]
        mini_batches = [
            batch_feed[i : i + batch_size]
            for i in range(0, len(batch_feed), batch_size)
        ]
        for idx, feed_batch in enumerate(mini_batches):
            feed_results = self.send_feed_batch(feed_batch=feed_batch)
            status_code_summary = Counter([x.status_code for x in feed_results])
            print(
                "Successful documents fed: {}/{}.\nBatch progress: {}/{}.".format(
                    status_code_summary[200], len(feed_batch), idx, len(mini_batches)
                )
            )
        return 0
