from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import logging
import tqdm
import sys

tracer = logging.getLogger('elasticsearch') 
tracer.setLevel(logging.CRITICAL) # supressing INFO messages for elastic-search

class ElasticSearch(object):
    
    def __init__(self, hostname, index_name, text_key, title_key, timeout=5000):
        logging.info("Activating Elastic Search....")
        self.index_name = index_name
        self.text_key = text_key
        self.title_key = title_key
        self.es = Elasticsearch([hostname], timeout=timeout, retry_on_timeout=True, maxsize=24)
        
    
    def create_index(self):
        """Create Elasticsearch Index

        Args:
            dims (int): dimension of dense vector
        """
        logging.info("Creating fresh Elasticsearch-Index named - {}".format(self.index_name))
        
        try:
            mapping = {
                "mappings" : {
                    "properties" : {
                        self.title_key: {"type": "text"},
                        self.text_key: {"type": "text"}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping, ignore=[400]) #400: IndexAlreadyExistsException
        except Exception as e:
            logging.error("Unable to create Index in Elastic Search. Reason: {}".format(e))
    
    def delete_index(self):
        """Delete Elasticsearch Index"""
        
        logging.info("Deleting previous Elasticsearch-Index named - {}".format(self.index_name))
        try:
            self.es.indices.delete(index=self.index_name, ignore=[400, 404]) # 404: IndexDoesntExistException
        except Exception as e:
            logging.error("Unable to create Index in Elastic Search. Reason: {}".format(e))
    
    def bulk_add_to_index(self, generate_actions, progress):
        """Bulk indexing to elastic search using generator actions

        Args:
            generate_actions (generator function): generator function must be provided 
            progress (tqdm.tqdm): tqdm progress_bar
        """
        for ok, action in streaming_bulk(
        client=self.es, index=self.index_name, actions=generate_actions,
        ):
            progress.update(1)
        progress.reset()
        progress.close()
    
    def lexical_multisearch(self, texts, top_hits, skip=0, text_present=False):
        """lexical search using text in Elastic Search

        Args:
            id (str): id of the head paragraph
            text (str): text of the head paragraph
            top_hits (int): top k hits to retrieved

        Returns:
            dict: Hit results
        """
        request = []
        
        assert top_hits + skip <= 10000, "Elastic-Search Window too large, Max-Size = 10000"
        
        if top_hits == 10000 and skip == 0 and text_present == False:
            top_hits = 9999
        
        for text in texts:
            req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
            req_body = {
                "_source": False, # No need to return source objects
                "query": {
                    "multi_match": { 
                        "query": text, # matching query with both text and title fields
                        "type": "best_fields",
                        "fields": [self.title_key, self.text_key],
                        "tie_breaker": 0.5
                        }
                    },
                "size": top_hits + skip + 1, # The same paragraph will occur in results
                }
            request.extend([req_head, req_body])
        
        res = self.es.msearch(body = request)

        result = []
        for resp in res["responses"]:
            if text_present:
                responses = resp["hits"]["hits"][(1+skip):]
            else:
                responses = resp["hits"]["hits"][skip:-1]
            
            hits = []
            for hit in responses:
                hits.append((hit["_id"], hit['_score']))

            result.append(self.hit_template(es_res=resp, hits=hits))
        return result
    
    
    def generate_actions(self, dictionary, update=False):
        """Iterator function for efficient addition to elastic-search
        Update - https://stackoverflow.com/questions/35182403/bulk-update-with-pythons-elasticsearch

        Args:
            dictionary (dict): dictionary with id and sentence_text/sentence_embedding.
        """
        for _id, value in dictionary.items():
            if not update: 
                doc = {
                    "_id": str(_id),
                    "_op_type": "index",
                    self.text_key: value[self.text_key],
                    self.title_key: value[self.title_key],
                }
            else:
                doc = {
                    "_id": str(_id),
                    "_op_type": "update",
                    "doc": {
                        self.text_key: value[self.text_key],
                        self.title_key: value[self.title_key],
                        }
                }
                
            yield doc
        
    def hit_template(self, es_res, hits):
        result = {
            'meta': {
                'total': es_res['hits']['total']['value'],
                'took': es_res['took'],
                'num_hits': len(hits)
            },
            'hits': hits,
        }
        return result