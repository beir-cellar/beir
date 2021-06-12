from pydantic import BaseSettings

class IndexSettings(BaseSettings):
    index_name: str = "beir/test"
    data_folder: str = "/home/datasets/"

def hit_template(hits):
    results = {}
    
    for qid, hit in hits.items():
        results[qid] = {}
        for i in range(0, len(hit)):
            results[qid][hit[i].docid] = hit[i].score
    
    return results