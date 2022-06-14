from typing import Dict, Tuple
import os
import logging
from datasets import load_dataset, Value, Features

logger = logging.getLogger(__name__)


class HFDataLoader:
    
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = "", streaming: bool = False, keep_in_memory: bool = False):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory
    
    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            # filter queries with no qrels
            qids = self.qrels['query-id']
            self.queries = self.queries.filter(lambda x: x['id'] in qids)
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", self.queries[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus
    
    def _load_corpus(self):
        corpus_ds = load_dataset('json', data_files=self.corpus_file, streaming=self.streaming, keep_in_memory=self.keep_in_memory)['train']
        corpus_ds = corpus_ds.cast_column('_id', Value('string'))
        corpus_ds = corpus_ds.rename_column('_id', 'id')
        corpus_ds = corpus_ds.remove_columns([col for col in corpus_ds.column_names if col not in ['id', 'text', 'title']])
        self.corpus = corpus_ds
    
    def _load_queries(self):
        queries_ds = load_dataset('json', data_files=self.query_file, keep_in_memory=self.keep_in_memory)['train']
        queries_ds = queries_ds.cast_column('_id', Value('string'))
        queries_ds = queries_ds.rename_column('_id', 'id')
        queries_ds = queries_ds.remove_columns([col for col in queries_ds.column_names if col not in ['id', 'text']])
        self.queries = queries_ds
        
    def _load_qrels(self):
        qrels_ds = load_dataset('csv', data_files=self.qrels_file, delimiter='\t', keep_in_memory=self.keep_in_memory)['train']
        features = Features({'query-id': Value('string'), 'corpus-id': Value('string'), 'score': Value('float')})
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds