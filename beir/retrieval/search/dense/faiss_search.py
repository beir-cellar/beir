import faiss
import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

#Parent class for any dense model
class DenseRetrievalFaissSearch:
    
    def __init__(self, model, n_clusters: int, nprobe: int, batch_size: int = 128, show_progress_bar: bool = True, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.embedding_size = 768
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.n_clusters = n_clusters
        self.quantizer = faiss.IndexFlatIP(self.embedding_size)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.embedding_size, self.n_clusters, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe
        self.results = {}
    
    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: List[int]) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid]["text"] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)
          
        logger.info("Encoding Whole Corpus... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]
        corpus_embeddings = self.model.encode_corpus(corpus, show_progress_bar=True, batch_size=self.batch_size)

        ### Create the FAISS index
        logger.info("Start creating FAISS index")
        # First, we need to normalize vectors to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]

        # Then we train the index to find a suitable clustering
        self.index.train(corpus_embeddings)

        # Finally we add all embeddings to the index
        self.index.add(corpus_embeddings)
            
        for query_itr in range(len(query_embeddings)):
            
            # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
            query_embedding = query_embeddings[query_itr]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = np.expand_dims(query_embedding, axis=0)

            # Search in FAISS. It returns a matrix with distances and corpus ids.
            distances, corpus = self.index.search(query_embedding, top_k)
            
            query_id = query_ids[query_itr]                  
            for (corpus_idx, score) in zip(corpus[0], distances[0]):
                corpus_id = corpus_ids[corpus_idx]
                if query_id != corpus_id:
                    self.results[query_id][corpus_id] = float(score)

        return self.results