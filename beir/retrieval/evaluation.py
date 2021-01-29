import pytrec_eval
import logging
from .search.dense import DenseRetrievalExactSearch as DRES

logger = logging.getLogger(__name__)
class EvaluateRetrieval:
    
    def __init__(self, model=None, k_values=[1,3,5,10,100,1000]):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.model = DRES(model) if model else model
            
    def retrieve(self, corpus, queries, qrels):
        if not self.model:
            raise ValueError("Model has not been provided!")
        return self.model.search(corpus, queries, self.top_k)
    
    def retrieve_and_evaluate(self, corpus, queries, qrels):
        results = self.retrieve(corpus, queries, qrels)
        return evaluate(qrels, results, self.k_values)
    
    @staticmethod
    def evaluate(qrels, results, k_values):
    
        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"ndcg@{k}"] = 0.0
            _map[f"map@{k}"] = 0.0
            recall[f"recall@{k}"] = 0.0
            precision[f"precision@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"ndcg@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"map@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"precision@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"ndcg@{k}"] = round(ndcg[f"ndcg@{k}"]/len(scores), 5)
            _map[f"map@{k}"] = round(_map[f"map@{k}"]/len(scores), 5)
            recall[f"recall@{k}"] = round(recall[f"recall@{k}"]/len(scores), 5)
            precision[f"precision@{k}"] = round(precision[f"precision@{k}"]/len(scores), 5)
        
        return ndcg, _map, recall, precision
    