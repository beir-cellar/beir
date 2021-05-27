import sys, os
import config

from fastapi import FastAPI, File, UploadFile
from pyserini.search import SimpleSearcher
from typing import Optional, List, Dict, Union

settings = config.IndexSettings()
app = FastAPI()

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = f'{dir_path}/datasets/{file.filename}'
    settings.data_folder = f'{dir_path}/datasets/'
    f = open(f'{filename}', 'wb')
    content = await file.read()
    f.write(content)
    return {"filename": file.filename}

@app.get("/index/")
def index(index_name: str, threads: Optional[int] = 8):
    settings.index_name = index_name

    command = f"python -m pyserini.index -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator -threads {threads} \
    -input {settings.data_folder} -index {settings.index_name} -storeRaw \
    -storePositions -storeDocvectors"
    
    os.system(command)
    
    return {200: "OK"}

@app.get("/lexical/search/")
def search(q: str, 
           k: Optional[int] = 1000,
           bm25: Optional[Dict[str, float]] = {"k1": 0.9, "b": 0.4},
           fields: Optional[Dict[str, float]] = {"contents": 1.0, "title": 1.0}):
    
    searcher = SimpleSearcher(settings.index_name)
    searcher.set_bm25(k1=bm25["k1"], b=bm25["b"])
    
    hits = searcher.search(q, k=k, fields=fields)
    results = []
    for i in range(0, len(hits)):
        results.append({'docid': hits[i].docid, 'score': hits[i].score})

    return {'results': results}

@app.post("/lexical/batch_search/")
def batch_search(queries: List[str],
                 qids: List[str],
                 k: Optional[int] = 1000, 
                 threads: Optional[int] = 8, 
                 bm25: Optional[Dict[str, float]] = {"k1": 0.9, "b": 0.4}, 
                 fields: Optional[Dict[str, float]] = {"contents": 1.0, "title": 1.0}):
    
    searcher = SimpleSearcher(settings.index_name)
    searcher.set_bm25(k1=bm25["k1"], b=bm25["b"])
    
    hits = searcher.batch_search(queries=queries, qids=qids, k=k, threads=threads, fields=fields)
    return {'results': config.hit_template(hits)}

@app.post("/lexical/rm3/batch_search/")
def batch_search_rm3(queries: List[str], 
                     qids: List[str], 
                     k: Optional[int] = 1000, 
                     threads: Optional[int] = 8, 
                     bm25: Optional[Dict[str, float]] = {"k1": 0.9, "b": 0.4}, 
                     fields: Optional[Dict[str, float]] = {"contents": 1.0, "title": 1.0},
                     rm3: Optional[Dict[str, Union[int, float]]] = {"fb_terms": 10, "fb_docs": 10, "original_query_weight": 0.5}):
    
    searcher = SimpleSearcher(settings.index_name)
    searcher.set_bm25(k1=bm25["k1"], b=bm25["b"])
    searcher.set_rm3(fb_terms=rm3["fb_terms"], fb_docs=rm3["fb_docs"], original_query_weight=rm3["original_query_weight"])
    
    hits = searcher.batch_search(queries=queries, qids=qids, k=k, threads=threads, fields=fields)
    return {'results': config.hit_template(hits)}