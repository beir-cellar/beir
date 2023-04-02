import unittest
from beir.retrieval.search.lexical.manticore_search import ManticoreLexicalSearch
from beir.retrieval.evaluation import EvaluateRetrieval


class TestManticoreLexicalSearch(unittest.TestCase):

    def setUp(self) -> None:
        self.application_name = "Manticore_test"
        self.corpus = {
            "1": {"title": "this is a title for query 1", "text": "this is a text for query 1"},
            "2": {"title": "this is a title for query 2", "text": "this is a text for query 2"},
            "3": {"title": "this is a title for query 3", "text": "this is a text for query 3"},
        }
        self.queries = {"1": "this is query 1", "2": "this is query 2"}


    def test_or_bm25(self):
        self.model = ManticoreLexicalSearch("test", 'http://localhost:9308')
        retriever = EvaluateRetrieval(self.model)
        results = retriever.retrieve(corpus=self.corpus, queries=self.queries)
        self.assertEqual(
            {"1", "2"}, 
            set( results.keys() )
            )
        for query_id in results.keys():
            self.assertEqual(
                {"1", "2", "3"}, 
                set( results[query_id].keys() ) 
            )
    

    def tearDown(self) -> None:
        self.model.clear()
        