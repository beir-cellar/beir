import sys, os
import config

from fastapi import FastAPI
from pyserini.search import SimpleSearcher
from typing import Optional, List

settings = config.IndexSettings()
app = FastAPI()

@app.get("/index/")
def index(data_folder: str, name: str, threads: Optional[int] = 8):
    settings.data_folder = data_folder
    settings.name = name
    command = f"python -m pyserini.index -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator -threads {threads} -input {data_folder} \
    -index {name} -storePositions -storeDocvectors -storeContents -storeRaw"
    os.system(command)
    return {200: "OK"}

@app.get("/search/")
def search(q: str, k: Optional[int] = 1000):
    searcher = SimpleSearcher(settings.name)
    hits = searcher.search(q, k=k)
    results = []
    for i in range(0, len(hits)):
        results.append({'docid': hits[i].docid, 'score': hits[i].score})

    return {'results': results}

@app.post("/batch_search/")
def batch_search(queries: List[str], qids: List[str], k: Optional[int] = 1000, threads: Optional[int] = 8):
    searcher = SimpleSearcher(settings.name)
    hits = searcher.batch_search(queries=queries[:10], qids=qids[:10], k=k, threads=threads)
    results = {}
    for qid, hit in hits.items():
        results[qid] = {}
        for i in range(0, len(hit)):
            results[qid][hit[i].docid] = hit[i].score

    return {'results': results}