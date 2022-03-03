import logging
from typing import List, Dict, Union, Tuple

def mrr(qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, Dict[str, float]], 
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    MRR = {}
    
    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"]/len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR

def recall_cap(qrels: Dict[str, Dict[str, int]], 
               results: Dict[str, Dict[str, float]], 
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    capped_recall = {}
    
    for k in k_values:
        capped_recall[f"R_cap@{k}"] = 0.0
    
    k_max = max(k_values)
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        for k in k_values:
            retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"] += (len(retrieved_docs) / denominator)

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = round(capped_recall[f"R_cap@{k}"]/len(qrels), 5)
        logging.info("R_cap@{}: {:.4f}".format(k, capped_recall[f"R_cap@{k}"]))

    return capped_recall


def hole(qrels: Dict[str, Dict[str, int]], 
               results: Dict[str, Dict[str, float]], 
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    Hole = {}
    
    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0
    
    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():    
            annotated_corpus.add(doc_id)
    
    k_max = max(k_values)
    logging.info("\n")
    
    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"]/len(qrels), 5)
        logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    return Hole

def top_k_accuracy(
        qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, Dict[str, float]], 
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    top_k_acc = {}
    
    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"]/len(qrels), 5)
        logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc