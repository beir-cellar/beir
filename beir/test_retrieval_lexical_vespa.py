import unittest
import shutil
import docker
from beir.retrieval.search.lexical.vespa_search import VespaLexicalSearch
from beir.retrieval.evaluation import EvaluateRetrieval


class TestVespaSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.application_name = "vespa_test"
        self.corpus = {
            "1": {"title": "this is a title 1", "text": "this is text 1"},
            "2": {"title": "this is a title 2", "text": "this is text 2"},
            "3": {"title": "this is a title 3", "text": "this is text 3"},
        }
        self.queries = {"1": "this is query 1", "2": "this is query 2"}

    def test_or_bm25(self):
        model = VespaLexicalSearch(application_name=self.application_name, initialize=True)
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus=self.corpus, queries=self.queries)
        self.assertEqual({"1", "2"}, set(results.keys()))
        for query_id in results.keys():
            self.assertEqual({"1", "2", "3"}, set(results[query_id].keys()))

    def test_or_native_rank(self):
        model = VespaLexicalSearch(
            application_name=self.application_name,
            initialize=True,
            match_phase="or",
            rank_phase="native_rank",
        )
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus=self.corpus, queries=self.queries)
        self.assertEqual({"1", "2"}, set(results.keys()))
        for query_id in results.keys():
            self.assertEqual({"1", "2", "3"}, set(results[query_id].keys()))

    def test_weakand_bm25(self):
        model = VespaLexicalSearch(
            application_name=self.application_name,
            initialize=True,
            match_phase="weak_and",
        )
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus=self.corpus, queries=self.queries)
        self.assertEqual({"1", "2"}, set(results.keys()))
        for query_id in results.keys():
            self.assertEqual({"1", "2", "3"}, set(results[query_id].keys()))

    def test_weakand_native_rank(self):
        model = VespaLexicalSearch(
            application_name=self.application_name,
            initialize=True,
            match_phase="weak_and",
            rank_phase="native_rank"
        )
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus=self.corpus, queries=self.queries)
        self.assertEqual({"1", "2"}, set(results.keys()))
        for query_id in results.keys():
            self.assertEqual({"1", "2", "3"}, set(results[query_id].keys()))

    def tearDown(self) -> None:
        shutil.rmtree(self.application_name, ignore_errors=True)
        client = docker.from_env()
        container = client.containers.get(self.application_name)
        container.stop()
        container.remove()
