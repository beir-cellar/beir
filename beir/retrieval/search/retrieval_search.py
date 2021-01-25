from sentence_transformers.util import pytorch_cos_sim
import torch

#Parent class for any dense model
class DenseRetrieval:
    
    def __init__(self, embedder, **kwargs):
        #embedder is class that provides encode_corpus() and encode_queries()
        self.embedder = embedder
        self.batch_size = 128
        self.corpus_chunk_size = 50000
        self.show_progress_bar = True
        self.results = {}
    
    def hnswlib_search(self, corpus, queries):
        pass
    
    def faiss_search(self, corpus, queries):
        pass
    
    def exact_search(self, corpus, queries, top_k):
        #Create embeddings for all queries using embedder.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
            
        print("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid]["text"] for qid in queries]
        query_embeddings = self.embedder.encode_queries(
            queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)
          
        print("Encoding Corpus...")
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]
            
        itr = range(0, len(corpus), self.corpus_chunk_size)
                
        for corpus_start_idx in itr:
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            #Encode chunk of corpus    
            sub_corpus_embeddings = self.embedder.encode_corpus(
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
