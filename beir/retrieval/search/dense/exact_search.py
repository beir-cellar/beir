from .util import pytorch_cos_sim
import logging
import sys
import torch

logger = logging.getLogger(__name__)

#Parent class for any dense model
class DenseRetrievalExactSearch:
    
    def __init__(self, model, batch_size=128, corpus_chunk_size=50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True #TODO: implement no progress bar if false
        self.results = {}
    
    def search(self, corpus, queries, top_k):
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid]["text"] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)
          
        logger.info("Encoding Corpus... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]
            
        itr = range(0, len(corpus), self.corpus_chunk_size)
                
        for corpus_start_idx in itr:
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            #Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                show_progress_bar=self.show_progress_bar, 
                batch_size=self.batch_size
                )

            #Compute cosine similarites
            cos_scores = pytorch_cos_sim(query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            #Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        self.results[query_id][corpus_id] = score
        
            return self.results 
