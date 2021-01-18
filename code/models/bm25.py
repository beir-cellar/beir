from elastic_search import ElasticSearch

class BM25:
    def __init__(self, hostname, index_name, text_key, title_key, timeout=100):
        print(hostname)
        self.es = ElasticSearch(hostname=hostname, index_name=index_name, 
                                text_key=text_key, title_key=title_key)
        self.initialise()
    
    def initialise(self):
        self.es.delete_index()
        self.es.create_index()
    
    def bm25_index(self, corpus, titles, text_key, title_key):
        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        # dictionary structure = {_id: {title_key: title, text_key: text}}
        dictionary = {idx: {
            title_key: titles.get(idx, None), 
            text_key: corpus.get(idx, None)
            } for idx in list(corpus.keys())
        }
        self.es.bulk_add_to_index(
                generate_actions=self.es.generate_actions(
                dictionary=dictionary, update=False),
                progress=progress
                )


    def bm25_rank(self, queries, judgements, top_k, batch_size):
        """[summary]

        Args:
            queries ([type]): [description]
            top_k ([type]): [description]
            batch_size ([type]): [description]
        """
        rank_results = {}
        
        _type = next(iter(list(judgements.values())[0]))
        generator = chunks(list(queries.keys()), batch_size)
        batches = int(len(queries)/batch_size)
        total = batches if len(queries) % batch_size == 0 else batches + 1 
        
        for query_id_chunks in tqdm.tqdm(generator, total=total):
            texts = [queries[query_id] for query_id in query_id_chunks]
            results = self.es.lexical_multisearch(
                texts=texts, top_hits=top_k + 1) 
            # add 1 extra just incase if query within document

            for (query_id, hit) in zip(query_id_chunks, results):
                scores = {}
                for corpus_id, score in hit['hits']:
                    corpus_id = type(_type)(corpus_id)
                    # making sure query doesnt return in results
                    if corpus_id != query_id:
                        scores[corpus_id] = score
                    
                rank_results[query_id] = scores
                
        return rank_results