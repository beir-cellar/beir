import pytrec_eval

from search import *
from models import *


class EvaluateRetrieval:
    
    def __init__(self, model, model_name, k_values=[1,3,5,10,100,1000]):
        self.k_values = k_values
        self.top_k = max(k_values)
        if model not in ["bm25"]:
            self.retrieval = DenseRetrieval(SentenceBERT(model_name))
    
    def evaluate(self, corpus, queries, qrels):
        
        results = self.retrieval.exact_search(corpus, queries, self.top_k)
        return self.rank(qrels, results, self.k_values)
    
    @staticmethod
    def rank(qrels, results, k_values):
    
        ndcg = {}
        map_ = {}
        recall = {}
        
        for k in k_values:
            ndcg[k] = 0.0
            map_[k] = 0.0
            recall[k] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[k] += scores[query_id]["ndcg_cut_" + str(k)]
                map_[k] += scores[query_id]["map_cut_" + str(k)]
                recall[k] += scores[query_id]["recall_" + str(k)]
        
        for k in k_values:
            ndcg[k] = round(ndcg[k]/len(scores), 5)
            map_[k] = round(map_[k]/len(scores), 5)
            recall[k] = round(recall[k]/len(scores), 5)
        
        return {"ndcg@k": ndcg, "map@k": map_, "recall@k": recall}
    