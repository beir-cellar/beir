import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

#Parent class for any reranking model
class Rerank:
    
    def __init__(self, model, batch_size: int = 128, **kwargs):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.rerank_results = {}
        
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])
            
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        rerank_scores = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]

        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score

        return self.rerank_results 
