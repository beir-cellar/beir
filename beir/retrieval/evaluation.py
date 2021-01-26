import pytrec_eval

from .search.dense import DenseRetrievalExactSearch
from .models import SentenceBERT, DPR


class EvaluateRetrieval:
    
    def __init__(self, model=None, model_name=None, k_values=[1,3,5,10,100,1000]):
        self.k_values = k_values
        self.top_k = max(k_values)
        
        if model.lower() in ["sbert", "sentence-bert", "sentence-transformer"]:
            self.retrieval = DenseRetrievalExactSearch(SentenceBERT(model_name))
        elif model.lower() in ["dpr", "dense-passage-retriever"]:
            self.retrieval = DenseRetrievalExactSearch(DPR())
    
    def retrieve(self, corpus, queries, qrels):
        return self.retrieval.search(corpus, queries, self.top_k)
    
    @staticmethod
    def evaluate(qrels, results, k_values):
    
        ndcg = {}
        _map = {}
        recall = {}
        
        for k in k_values:
            ndcg[f"ndcg@{k}"] = 0.0
            _map[f"map@{k}"] = 0.0
            recall[f"recall@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"ndcg@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"map@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"recall@{k}"] += scores[query_id]["recall_" + str(k)]
        
        for k in k_values:
            ndcg[f"ndcg@{k}"] = round(ndcg[f"ndcg@{k}"]/len(scores), 5)
            _map[f"map@{k}"] = round(_map[f"map@{k}"]/len(scores), 5)
            recall[f"recall@{k}"] = round(recall[f"recall@{k}"]/len(scores), 5)
        
        return ndcg, _map, recall
    