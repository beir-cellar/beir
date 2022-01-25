import shutil
from typing import Dict, Optional
from vespa.application import Vespa
from vespa.package import ApplicationPackage, Field, FieldSet, RankProfile, QueryField
from vespa.query import QueryModel, OR, RankProfile as Ranking, WeakAnd
from vespa.deployment import VespaDocker


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
        self.application_name = application_name
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
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
            Field(
                name="body",
                type="string",
                indexing=["index", "summary"],
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
            self.vespa_docker.container.stop()  # stop docker container
            self.vespa_docker.container.remove()  # stop docker container

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

        if self.match_phase == "or":
            match_phase = OR()
        elif self.match_phase == "weak_and":
            match_phase = WeakAnd(hits=top_k)
        else:
            ValueError("'match_phase' should be either 'or' or 'weak_and'")

        if self.rank_phase not in ["bm25", "native_rank"]:
            ValueError("'rank_phase' should be either 'bm25' or 'native_rank'")

        query_model = QueryModel(
            name=self.match_phase + "_bm25",
            match_phase=match_phase,
            rank_profile=Ranking(name=self.rank_phase, list_features=False),
        )

        for idx, (query_id, query) in enumerate(zip(query_ids, queries)):
            if (idx % 1000) == 0:
                print(
                    "{}, {}, {}: {}/{}".format(
                        self.application_name,
                        match_phase,
                        self.rank_phase,
                        idx,
                        len(query_ids),
                    )
                )
            scores = {}
            query_result = self.app.query(
                query=query, query_model=query_model, hits=top_k, timeout="10 s"
            )
            for hit in query_result.hits:
                scores[hit["fields"]["id"]] = hit["relevance"]
            self.results[query_id] = scores
        return self.results

    def index(self, corpus: Dict[str, Dict[str, str]]):
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
        return self.app.feed_batch(batch=batch_feed)
