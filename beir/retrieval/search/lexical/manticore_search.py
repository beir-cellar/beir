import time
import manticoresearch
from json import dumps 
from tqdm import tqdm
from typing import Dict
from wasabi import msg
from urllib.parse import quote


class ManticoreLexicalSearch:
    
    ESC_CHARS = ['\\', "'", '!', '"', '$', '(', ')', '-', '/', '<', '@', '^', '|', '~', ]
    
    def __init__(
        self,
        index_name: str,
        host: str,
        store_indexes: bool = False
    ):
        self.store_indexes = store_indexes
        # Escape special characters in index name
        for ch in self.ESC_CHARS:
            index_name = index_name.replace(ch, '_') 
        self.index_name = "beir_benchmark_" + index_name
        # Initialize Manticore instance and create benchmark index
        with manticoresearch.ApiClient( manticoresearch.Configuration(host=host) ) as api_client:
            self.__index_api = manticoresearch.IndexApi(api_client)
            self.__utils_api = manticoresearch.UtilsApi(api_client)
        body = quote("CREATE TABLE IF NOT EXISTS " + self.index_name +
            "(_id string, title text, body text) stopwords='en' stopwords_unstemmed='1'" + 
            " html_strip='1' morphology='lemmatize_en_all' index_exact_words='1' index_field_lengths='1' ")
        self.__utils_api.sql(body)
            
    
    def clear(self):
        # Clear existing benchmark index
        self.__utils_api.sql( quote("DROP TABLE IF EXISTS " + self.index_name) )
    
    
    def __index_exists(self) -> bool:
        req = "SELECT 1 FROM " + self.index_name + " LIMIT 1"
        resp = self.__utils_api.sql(req, raw_response=False)
        return True if resp['hits']['hits'] else False


    def __prepare_query(self, query:str) -> str:
        # Escape necessary characters and convert query to 'or' search mode
        for ch in self.ESC_CHARS:
            if ch == "'":
                repl = '\\'
            elif ch == '\\':
                repl = '\\\\\\'
            else:
                repl = '\\\\'
            query = query.replace(ch, repl + ch )
        if query[-1] == '=':
            query = query[0:-1] + '\\\\='
        return '"{}"/1'.format(query)
    
    
    def index(self, corpus: Dict[str, Dict[str, str]], batch_size: int = 10000):
        msg.info("Indexing:")
        docs = list( corpus.items() )
        for i in range(0, len(corpus), batch_size):
            index_docs = [ {
                "insert": {
                    "index": self.index_name,
                    "doc": {
                        "_id": str(id),
                        "title": doc["title"],
                        "body": doc["text"],
                    }
                }
            } for id,doc in docs[i:i + batch_size] ]
            msg.info( "Batch {} with {} docs".format(i//batch_size+1, len(index_docs) ) )
            self.__index_api.bulk( '\n'.join( map(dumps, index_docs) ) )
        self.__utils_api.sql("FLUSH RAMCHUNK " + self.index_name)
        time.sleep(5)
        self.__utils_api.sql( quote("OPTIMIZE INDEX " + self.index_name + " OPTION cutoff=1, sync=1; ") )

    
    def search(
        self, 
        corpus: Dict[str, Dict[str, str]], 
        queries: Dict[str, str], 
        top_k: int,
        *args,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        req_tmpl = "SELECT _id, WEIGHT() as w FROM " + self.index_name + " WHERE MATCH('@(title,body){}') " \
        "OPTION ranker=expr('10000 * bm25f(1.2,0.75)'), idf='plain,tfidf_unnormalized', max_matches=" \
         + str(10)
        if not self.__index_exists():
            self.index(corpus)
        msg.info("Evaluating:")
        for qid, query in tqdm( queries.items() ):
            req = req_tmpl.format( self.__prepare_query(query) )
            resp = self.__utils_api.sql(req, raw_response=False)
            query_docs = { doc['_source']['_id']:doc['_source']['w']
                          for doc in resp['hits']['hits'] if doc['_source']['w'] }
            if query_docs:
                results.update( { qid: query_docs  } )
        if not self.store_indexes:
            self.clear()
        return results

