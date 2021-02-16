from tqdm.autonotebook import trange
from ..util import write_to_json, write_to_tsv
from typing import Dict
import logging, os

logger = logging.getLogger(__name__)

class QueryFilter:
    def __init__(self, model, **kwargs):
        self.model = model
        self.filtered_qrels = {}
        self.filtered_queries = {}

    @staticmethod
    def save(output_dir: str, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], prefix: str):
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, prefix + "-qrels"), exist_ok=True)
        
        query_file = os.path.join(output_dir, prefix + "-queries.jsonl")
        qrels_file = os.path.join(output_dir, prefix + "-qrels", "train.tsv")
        
        logger.info("Saving Filtered Queries to {}".format(query_file))        
        write_to_json(output_file=query_file, data=queries)

        logger.info("Saving Filtered Qrels to {}".format(qrels_file))
        write_to_tsv(output_file=qrels_file, data=qrels)

    def filter(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               qrels: Dict[str, Dict[str, int]], 
               output_dir: str, 
               threshold: int = 0.5, 
               batch_size: int = 32, 
               prefix: str = "filter"):
        
        logger.info("Starting to Filter Questions...")
        logger.info("Batch Size: --- {} ---".format(batch_size))
        
        pairs = []
        for query_id in qrels.keys():
            for doc_id, score in qrels[query_id].items():
                pairs.append([query_id, doc_id])
        
        for start_idx in trange(0, len(pairs), batch_size, desc='pas'):
            sentences = [[queries[pair[0]], corpus[pair[1]]["title"] + " " + corpus[pair[1]]["text"]] for pair in pairs[start_idx:start_idx + batch_size]]
            scores = self.model.predict(sentences, batch_size)
            for pair, score in zip(pairs, scores):
                if score >= threshold:
                    self.filtered_queries[pair[0]] = queries[pair[0]]
                    self.filtered_qrels[pair[0]] = {pair[1]: 1}
        
        # Saving finally all the questions
        logger.info("Saving {} Filtered Queries...".format(len(self.filtered_qrels)))
        self.save(output_dir, self.filtered_queries, self.filtered_qrels, prefix)